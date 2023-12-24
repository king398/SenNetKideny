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
        if not inference:
            self.unet.encoder.model.set_grad_checkpointing(True)

    def forward(self, x):
        # Pad the input
        original_size = x.shape[2:]
        x, pad = self._pad_image(x)

        # Forward pass through Unet
        x = self.unet(x)

        # Remove padding
        x = self._unpad(x, original_size, pad)
        return x

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 384):
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_factor - h % pad_factor) % pad_factor
        w_pad = (pad_factor - w % pad_factor) % pad_factor

        # Calculate padding
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        return x, pad

    def _unpad(self, x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]
