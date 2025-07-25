import logging
import torch.nn as nn
import numpy as np
from utils.torchsummary import summary
from typing import Any, Dict


class BaseModel(nn.Module):
    """
    Base class for all model architectures.

    This class provides common functionality for all models including
    parameter counting and logging utilities.
    """

    def __init__(self) -> None:
        """Initialize the base model with a logger."""
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self) -> Any:
        """
        Forward pass of the model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def summary(self) -> None:
        """Print a summary of the model including the number of trainable parameters."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"Nbr of trainable parameters: {nbr_params}")

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            str: Model string representation with parameter count.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super(BaseModel, self).__str__()
            + f"\nNbr of trainable parameters: {nbr_params}"
        )
        # return summary(self, input_shape=(2, 3, 224, 224))
