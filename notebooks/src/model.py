import segmentation_models_pytorch as smp
from torch.nn import Module

import segmentation_models_pytorch as smp
from torch.nn import Module
import torch
import math

# Example usage
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int):
        super(ReturnModel, self).__init__()
        # Initialize the Unet model
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        # Forward pass through Unet
        x = self.unet(x)
        # Upscale the output
        return x
