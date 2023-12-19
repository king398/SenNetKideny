import segmentation_models_pytorch as smp
from torch.nn import Module


import segmentation_models_pytorch as smp
from torch.nn import Module
import torch
import math

"""class return_model(Module):
    def __init__(self, model_name: str, in_channels: int, classes: int):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        # Padding the input
        padded_x, original_shape = self.pad_to_factor(x)
        # Forward pass through the model
        output = self.model(padded_x)
        # Returning to original shape
        return self.unpad_to_original_shape(output, original_shape)

    def pad_to_factor(self, x, factor=32):
  
b, c, h, w = x.shape
new_h = math.ceil(h / factor) * factor
new_w = math.ceil(w / factor) * factor
pad_h = new_h - h
pad_w = new_w - w
# Padding format: (left, right, top, bottom)
padded_x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
return padded_x, (h, w)

def unpad_to_original_shape(self, x, original_shape):

original_h, original_w = original_shape
return x[:, :, 0:original_h, 0:original_w]"""

# Example usage
def return_model(model_name: str, in_channels: int, classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,




    )
    return model