import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, Optional


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(
        self, val: Union[float, np.ndarray], weight: Union[float, int]
    ) -> None:
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(
        self, val: Union[float, np.ndarray], weight: Union[float, int] = 1
    ) -> None:
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val: Union[float, np.ndarray], weight: Union[float, int]) -> None:
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))  # type: ignore
        self.count = self.count + weight  # type: ignore
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)  # type: ignore


def batch_pix_accuracy(predict, target, labeled):
    """
    Compute pixel-wise accuracy for a batch of predictions.

    Args:
        predict (torch.Tensor): Predicted labels
        target (torch.Tensor): Ground truth labels
        labeled (torch.Tensor): Mask indicating valid pixels

    Returns:
        tuple: (pixel_correct, pixel_labeled) as numpy arrays
    """
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(predict, target, num_class, labeled):
    """
    Compute intersection and union for IoU calculation across a batch.

    Args:
        predict (torch.Tensor): Predicted labels
        target (torch.Tensor): Ground truth labels
        num_class (int): Number of classes
        labeled (torch.Tensor): Mask indicating valid pixels

    Returns:
        tuple: (area_inter, area_union) as numpy arrays
    """
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (
        area_inter <= area_union
    ).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_class):
    """
    Evaluate segmentation metrics including pixel accuracy and IoU.

    Args:
        output (torch.Tensor): Model output predictions
        target (torch.Tensor): Ground truth labels
        num_class (int): Number of classes

    Returns:
        list: [correct_pixels, total_labeled_pixels, intersection, union]
    """
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)
    return [
        np.round(correct, 5),
        np.round(num_labeled, 5),
        np.round(inter, 5),
        np.round(union, 5),
    ]
