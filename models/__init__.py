"""
Deep learning models for plant damage estimation.

This module contains various model architectures including U-Net variants
with different backbones (VGG16, ResNet) and self-supervised learning heads.
"""

from .unet import UNet, UNetResnet, UNetVGG16

__all__ = ['UNet', 'UNetResnet', 'UNetVGG16']
