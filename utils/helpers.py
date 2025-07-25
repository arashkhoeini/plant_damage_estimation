import os
import torch
import torch.nn as nn
import numpy as np
import math
import os.path as osp
from typing import Union, Tuple, List, Any
from PIL import Image


def dir_exists(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path (str): Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def initialize_weights(*models: nn.Module) -> None:
    """
    Initialize weights for neural network models using standard initialization schemes.

    Args:
        *models: Variable number of PyTorch models to initialize
    """
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def get_upsampling_weight(
    in_channels: int, out_channels: int, kernel_size: int
) -> torch.Tensor:
    """
    Generate bilinear upsampling weights for transposed convolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel

    Returns:
        torch.Tensor: Upsampling weight tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros(
        (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64
    )
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def colorize_mask(mask: np.ndarray, palette: List[int]) -> Image.Image:
    """
    Apply color palette to a segmentation mask.

    Args:
        mask (numpy.ndarray): Segmentation mask
        palette (list): Color palette as a list of RGB values

    Returns:
        PIL.Image: Colorized mask image
    """
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    # new_mask.putpalette(palette)
    return new_mask


def set_trainable_attr(m, b):
    """
    Set trainable attribute for a model and freeze/unfreeze its parameters.

    Args:
        m (nn.Module): PyTorch model
        b (bool): Whether to make the model trainable
    """
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    """
    Apply a function to leaf modules (modules with no children).

    Args:
        m (nn.Module or list): Model or list of modules
        f (function): Function to apply to each leaf module
    """
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def mkdir(p):
    if not osp.exists(p):
        os.makedirs(p)
        print("DIR {} created".format(p))
    return p
