from models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from models import resnet
from torchvision.models import vgg16_bn
import time


def x2conv(in_channels, out_channels, inner_channels=None):
    """
    Create a double convolution block with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        inner_channels (int, optional): Number of channels in the first conv layer.
                                      Defaults to out_channels // 2

    Returns:
        nn.Sequential: Double convolution block
    """
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return down_conv


class encoder(nn.Module):
    """
    Encoder block for U-Net consisting of double convolution followed by max pooling.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor after convolution and pooling
        """
        x = self.down_conv(x)
        x = self.pool(x)
        return x


class decoder(nn.Module):
    """
    Decoder block for U-Net consisting of upsampling and double convolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        """
        Forward pass of the decoder block.

        Args:
            x_copy (torch.Tensor): Skip connection tensor from encoder
            x (torch.Tensor): Input tensor from previous layer
            interpolate (bool): Whether to use interpolation for size matching

        Returns:
            torch.Tensor: Output tensor after upsampling and convolution
        """
        x = self.up(x)

        if (x.size(2) != x_copy.size(2)) or (x.size(3) != x_copy.size(3)):
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(
                    x,
                    size=(x_copy.size(2), x_copy.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                # Padding in case the incomping volumes are of different sizes
                diffY = x_copy.size()[2] - x.size()[2]
                diffX = x_copy.size()[3] - x.size()[3]
                x = F.pad(
                    x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
                )

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(BaseModel):
    """
    Standard U-Net architecture for semantic segmentation.

    This implementation uses a symmetric encoder-decoder structure with skip connections.

    Args:
        num_classes (int): Number of output classes for segmentation
        in_channels (int): Number of input channels (default: 3 for RGB)
        freeze_bn (bool): Whether to freeze batch normalization layers
    """

    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()

        self.start_conv = x2conv(in_channels, 64)
        self.down1 = encoder(64, 128)
        self.down2 = encoder(128, 256)
        self.down3 = encoder(256, 512)
        self.down4 = encoder(512, 1024)

        self.middle_conv = x2conv(1024, 1024)

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output segmentation map of shape (N, num_classes, H, W)
        """
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        return x

    def get_encoder_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


"""
-> Unet with a resnet backbone
"""


class UNetResnet(BaseModel):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        backbone="resnet50",
        pretrained=True,
        freeze_bn=False,
        freeze_backbone=False,
        **_
    ):
        super(UNetResnet, self).__init__()
        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)

        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 = nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)

        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone:
            set_trainable(
                [self.initial, self.layer1, self.layer2, self.layer3, self.layer4],
                False,
            )

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))  # type: ignore
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(
            x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True
        )
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(
            x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True
        )
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(
            x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True
        )
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(
            self.initial.parameters(),  # type: ignore
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
        )

    def get_encoder_params(self):
        return chain(
            self.conv1.parameters(),
            self.upconv1.parameters(),
            self.conv2.parameters(),
            self.upconv2.parameters(),
            self.conv3.parameters(),
            self.upconv3.parameters(),
            self.conv4.parameters(),
            self.upconv4.parameters(),
            self.conv5.parameters(),
            self.upconv5.parameters(),
            self.conv6.parameters(),
            self.conv7.parameters(),
        )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )


class UNetVGG16(BaseModel):
    def __init__(self, n_classes, pretrained, encoder=None):
        super().__init__()

        if encoder is None:
            self.encoder = vgg16_bn(pretrained=pretrained).features
        else:
            self.encoder = encoder.features

        self.block1 = nn.Sequential(*self.encoder[:6])  # type: ignore
        self.block2 = nn.Sequential(*self.encoder[6:13])  # type: ignore
        self.block3 = nn.Sequential(*self.encoder[13:23])  # type: ignore
        self.block4 = nn.Sequential(*self.encoder[23:33])  # type: ignore

        self.bottleneck = nn.Sequential(*self.encoder[33:])  # type: ignore
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 256, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 128, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 64, 32)
        self.conv_final = nn.Conv2d(32, n_classes, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.projection_head = nn.Sequential(
            nn.Linear(25088, 4096), nn.Linear(4096, 512)
        )

        self.prediction_head = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512))

    def forward(self, x, pretraining=False):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # inference_start = time.perf_counter()
        bottleneck = self.bottleneck(block4)

        if not pretraining:
            x = self.conv_bottleneck(bottleneck)
            x = self.up_conv6(x)
            x = F.interpolate(
                x,
                size=(block4.size(2), block4.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            x = torch.cat([x, block4], dim=1)
            x = self.conv6(x)
            x = self.up_conv7(x)
            x = F.interpolate(
                x,
                size=(block3.size(2), block3.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            x = torch.cat([x, block3], dim=1)
            x = self.conv7(x)
            x = self.up_conv8(x)
            x = F.interpolate(
                x,
                size=(block2.size(2), block2.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            x = torch.cat([x, block2], dim=1)
            x = self.conv8(x)
            x = self.up_conv9(x)
            x = F.interpolate(
                x,
                size=(block1.size(2), block1.size(3)),
                mode="bilinear",
                align_corners=True,
            )
            x = torch.cat([x, block1], dim=1)
            x = self.conv9(x)
            x = self.conv_final(x)
            return x
        else:
            avg = self.avgpool(bottleneck)
            avg = torch.flatten(avg, 1)
            z = self.projection_head(avg)
            p = self.prediction_head(z)
            return z, p

    def get_encoder_params(self):
        return chain(
            self.block1.parameters(),
            self.block2.parameters(),
            self.block3.parameters(),
            self.block4.parameters(),
            self.bottleneck.parameters(),
            self.conv_bottleneck.parameters(),
        )

    def get_decoder_params(self):
        return chain(
            self.up_conv6.parameters(),
            self.conv6.parameters(),
            self.up_conv7.parameters(),
            self.conv7.parameters(),
            self.up_conv8.parameters(),
            self.conv8.parameters(),
            self.up_conv9.parameters(),
            self.conv9.parameters(),
            self.conv_final.parameters(),
        )


class UNetVGG16MoCo(UNetVGG16):

    def __init__(
        self,
        n_classes,
        encoder=None,
        queue_size=8192,
        momentum=0.999,
        T=0.07,
        pretrained=False,
    ):
        super().__init__(n_classes, pretrained, encoder)

        self.momentum = momentum
        self.T = T
        self.queue_size = queue_size
        enc = vgg16_bn(pretrained=pretrained).features
        self.target_encoder = torch.nn.Sequential(
            nn.Sequential(*enc[:6]),  # type: ignore
            nn.Sequential(*enc[6:13]),  # type: ignore
            nn.Sequential(*enc[13:23]),  # type: ignore
            nn.Sequential(*enc[23:33]),  # type: ignore
            self.avgpool,
            MyFlatten(),
            nn.Sequential(nn.Linear(25088, 4096), nn.Linear(4096, 512)),
        )

        # create the queue
        self.register_buffer("queue", torch.randn(512, queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x1, x2=None, pretraining=False):
        x1 = x1.to(torch.float32)

        if pretraining:
            z1, _ = super().forward(x1, pretraining)
            z1 = torch.nn.functional.normalize(z1, dim=1)
            x2 = x2.to(torch.float32)  # type: ignore
            with torch.no_grad():
                z2 = self.target_encoder(x2)
                z2 = torch.nn.functional.normalize(z2, dim=1)
            preds_raw = None

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum("nc,nc->n", [z1, z2]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [z1, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(
                logits.shape[0], dtype=torch.long, device=logits.device
            )

            # dequeue and enqueue
            self._dequeue_and_enqueue(z2)
            return logits, labels
        else:
            return super().forward(x1, pretraining)

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        for param_q, param_k in zip(
            chain(self.encoder.parameters(), self.pretraining_head.parameters()),  # type: ignore
            self.target_encoder.parameters(),
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        ptr = int(self.queue_ptr)  # type: ignore

        # replace the keys at ptr (dequeue and enqueue)
        offset = ptr + keys.shape[0]
        if offset >= self.queue_size:
            self.queue[:, ptr:] = keys.T[:, : self.queue_size - ptr]
            self.queue[:, : offset - self.queue_size] = keys.T[
                :, self.queue_size - ptr :
            ]
            ptr = offset - self.queue_size
        else:
            self.queue[:, ptr:offset] = keys.T
            ptr = (ptr + keys.shape[0]) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr  # type: ignore


class UNetVGG16PxCL(BaseModel):
    def __init__(self, n_classes, pretrained=False, encoder=None):
        super().__init__()

        if encoder is None:
            self.encoder = vgg16_bn(pretrained=pretrained).features
        else:
            self.encoder = encoder.features

        self.block1 = nn.Sequential(*self.encoder[:6])  # type: ignore
        self.block2 = nn.Sequential(*self.encoder[6:13])  # type: ignore
        self.block3 = nn.Sequential(*self.encoder[13:23])  # type: ignore
        self.block4 = nn.Sequential(*self.encoder[23:33])  # type: ignore

        self.bottleneck = nn.Sequential(*self.encoder[33:])  # type: ignore
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 256, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 128, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 64, 32)
        self.conv_final = nn.Conv2d(32, n_classes, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.projection_head = nn.Sequential(
            conv(32, 32),
            conv(32, 32),
        )

    def forward(self, x, pretraining=False):
        x = x.to(torch.float32)

        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # inference_start = time.perf_counter()
        bottleneck = self.bottleneck(block4)
        x = self.conv_bottleneck(bottleneck)
        x = self.up_conv6(x)
        x = F.interpolate(
            x,
            size=(block4.size(2), block4.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)
        x = self.up_conv7(x)
        x = F.interpolate(
            x,
            size=(block3.size(2), block3.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)
        x = self.up_conv8(x)
        x = F.interpolate(
            x,
            size=(block2.size(2), block2.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)
        x = self.up_conv9(x)
        x = F.interpolate(
            x,
            size=(block1.size(2), block1.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)
        if not pretraining:
            x = self.conv_final(x)
            return x
        else:
            z = self.projection_head(x)
            return z

    def get_encoder_params(self):
        return chain(
            self.block1.parameters(),
            self.block2.parameters(),
            self.block3.parameters(),
            self.block4.parameters(),
            self.bottleneck.parameters(),
            self.conv_bottleneck.parameters(),
        )

    def get_decoder_params(self):
        return chain(
            self.up_conv6.parameters(),
            self.conv6.parameters(),
            self.up_conv7.parameters(),
            self.conv7.parameters(),
            self.up_conv8.parameters(),
            self.conv8.parameters(),
            self.up_conv9.parameters(),
            self.conv9.parameters(),
            self.conv_final.parameters(),
        )


class MyFlatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        flat = torch.flatten(x, 1)
        return flat
