import segmentation_models_pytorch as smp
from torch.nn import Module

import segmentation_models_pytorch as smp
from torch.nn import Module
import torch
import math




# Example usage
def return_model(model_name: str, in_channels: int, classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
        encoder_depth=4,
        decoder_channels=[256, 128, 64, 32],

    )
    return model
