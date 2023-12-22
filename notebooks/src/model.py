import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, inference: bool = False):
        super(ReturnModel, self).__init__()
        # Initialize the Unet model
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        # Pad the input
        original_size = x.shape[2:]
        x, pad = self._pad_to_32(x)

        # Forward pass through Unet
        x = self.unet(x)

        # Remove padding
        x = self._unpad(x, original_size, pad)
        return x

    def _pad_to_32(self, x):
        h, w = x.shape[2], x.shape[3]
        h_pad = (32 - h % 32) % 32
        w_pad = (32 - w % 32) % 32

        # Calculate padding
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        return x, pad

    def _unpad(self, x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]
