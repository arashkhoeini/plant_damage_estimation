from typing import Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight
from utils.lovasz_losses import lovasz_softmax
from random import random


def make_one_hot(labels: torch.Tensor, classes: int) -> torch.Tensor:
    """
    Convert class labels to one-hot encoding.

    Args:
        labels (torch.Tensor): Class labels tensor
        classes (int): Number of classes

    Returns:
        torch.Tensor: One-hot encoded tensor
    """
    one_hot = (
        torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3])
        .zero_()
        .to(labels.device)
    )
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for balanced training.

    Args:
        target (torch.Tensor): Target labels tensor

    Returns:
        torch.Tensor: Class weights tensor
    """
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()


class CrossEntropyLoss2d(nn.Module):
    """
    2D Cross Entropy Loss for semantic segmentation.

    A wrapper around PyTorch's CrossEntropyLoss for 2D segmentation tasks.

    Args:
        weight (torch.Tensor, optional): Manual rescaling weight for each class
        ignore_index (int): Index to ignore in loss computation (default: 255)
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, weight=None, ignore_index=255, reduction="mean"):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    def forward(self, output, target):
        """
        Forward pass of the loss function.

        Args:
            output (torch.Tensor): Model predictions of shape (N, C, H, W)
            target (torch.Tensor): Ground truth labels of shape (N, H, W)

        Returns:
            torch.Tensor: Computed loss value
        """
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - (
            (2.0 * intersection + self.smooth)
            / (output_flat.sum() + target_flat.sum() + self.smooth)
        )
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(
            reduce=False, ignore_index=ignore_index, weight=alpha
        )

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction="mean", ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes="present", per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class SimSiam(nn.Module):
    def __init__(self, version="simplified") -> None:
        super().__init__()
        self.version = version

    def loss_fn(self, z, p):
        if self.version == "original":
            z = z.detach()
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()

        elif self.version == "simplified":
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

    def forward(self, z1, p1, z2, p2):
        return self.loss_fn(z2, p1) / 2 + self.loss_fn(z1, p2) / 2


class MoCo(nn.CrossEntropyLoss):

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction)


class PxCL(nn.Module):

    def __init__(self, temperature=0.07) -> None:
        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2, labels, chunks=64):

        batch_size = z1.size(0)
        z1_norm = torch.flatten(F.normalize(z1, p=2, dim=1), 2)
        z2_norm = torch.flatten(F.normalize(z2, p=2, dim=1), 2)
        labels = torch.flatten(labels, 1)

        contrastive_loss = []
        for b in range(batch_size):
            partitions1, partitions2 = [], []
            for chunk in range(chunks):
                # print('\n', int(chunk/chunks*z1_norm.size(2)) , int((chunk+1)/chunks*z1_norm.size(2)) )
                partitions1.append(
                    (
                        z1_norm[
                            b,
                            :,
                            int(chunk / chunks * z1_norm.size(2)) : int(
                                (chunk + 1) / chunks * z1_norm.size(2)
                            ),
                        ],
                        labels[
                            b,
                            int(chunk / chunks * z1_norm.size(2)) : int(
                                (chunk + 1) / chunks * z1_norm.size(2)
                            ),
                        ],
                    )
                )
                partitions2.append(
                    (
                        z2_norm[
                            b,
                            :,
                            int(chunk / chunks * z2_norm.size(2)) : int(
                                (chunk + 1) / chunks * z2_norm.size(2)
                            ),
                        ],
                        labels[
                            b,
                            int(chunk / chunks * z2_norm.size(2)) : int(
                                (chunk + 1) / chunks * z2_norm.size(2)
                            ),
                        ],
                    )
                )
            for i, (reps1, l1) in enumerate(partitions1):
                negative_chunk = None
                for j, (reps2, l2) in enumerate(partitions2):
                    if i == j:
                        # compute similarities
                        sims = F.cosine_similarity(reps1, reps2, dim=0)
                    else:
                        if negative_chunk is None:
                            negative_chunk = j
                        else:
                            if random() > 0.5:
                                negative_chunk = j

                # compute pixel-level contrastive loss for this chunk
                positive_logits = sims / self.temperature
                mask = ~(l1.unsqueeze(1) == l2.unsqueeze(0))
                negative_logits = (
                    torch.mm(reps1.transpose(0, 1), reps2) * mask / self.temperature
                )
                # print(positive_logits.view(-1,1).shape, negative_logits.shape)
                logits = torch.cat(
                    (positive_logits.view(-1, 1), negative_logits), dim=1  # type: ignore
                )
                log_probabilities = F.log_softmax(logits, dim=1)
                # print('final shape:', log_probabilities.shape)
                contrastive_loss.append(-log_probabilities.mean())

        return sum(contrastive_loss) / len(contrastive_loss)

        # two for loops. in outer loop I iterate over images in a batch.
        # and in the innter for loop I iterate over chunks of the image
        # average over all the contrastive losses.
        # dot_product = torch.matmul(z1_norm.transpose(1, 2), z2_norm)
        # for b in range(batch_size):
        #     partitions1, partitions2 = [], []
        #     for chunk in range(chunks):
        #         print('\n', int(chunk/chunks*z1_norm.size(2)) , int((chunk+1)/chunks*z1_norm.size(2)) )
        #         partitions1.append((z1_norm[b, :, int(chunk/chunks*z1_norm.size(2)) : int((chunk+1)/chunks*z1_norm.size(2)) ],
        #                             labels1[b, int(chunk/chunks*z1_norm.size(2)) : int((chunk+1)/chunks*z1_norm.size(2)) ]))
        #         partitions2.append((z2_norm[b, :, int(chunk/chunks*z2_norm.size(2)) : int((chunk+1)/chunks*z2_norm.size(2)) ],
        #                             labels2[b, int(chunk/chunks*z2_norm.size(2)) : int((chunk+1)/chunks*z2_norm.size(2))]))
        #     for rep1, l1 in partitions1:
        #         for rep2, l2 in partitions2:
        #             print('Rep1 shapes: ')
        #             print(rep1.shape, l1.shape)
        #             dot_product = torch.matmul(rep1.transpose(0, 1), rep2)
        #             mask = ~(l1.unsqueeze(1) == l2.unsqueeze(0))
        #             print(dot_product.shape)
        #             print(mask.shape)

        #

        # diag_mask = torch.eye(2 * batch_size, device=z1.device)
        # positive_indices = diag_mask.byte().nonzero()[:, 0]
        # log_prob_positive = log_probabilities.view(-1)[positive_indices]

        # # Calculate the contrastive loss using InfoNCE
        # contrastive_loss = -log_prob_positive.mean()

        # return contrastive_loss
