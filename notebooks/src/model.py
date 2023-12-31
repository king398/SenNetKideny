import segmentation_models_pytorch as smp
import timm
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import TimmUniversalEncoder
from torch.utils.checkpoint import checkpoint
from nextvit import *
from segmentation_models_pytorch.base.heads import SegmentationHead
from torch import nn
import torch


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
        # if not inference:
        #    self.unet.encoder.model.set_grad_checkpointing(True)
        self.inference = inference

    def forward(self, x, inference: bool = False):
        # Pad the input
        original_size = x.shape[2:]
        x, pad = self._pad_image(x)

        # Forward pass through Unet
        x = checkpoint(self.unet.encoder, x, )

        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)
        # Remove padding
        x = self._unpad(x, original_size, pad)

        return x

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 224):
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


class ReturnModelNextVit(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, inference: bool = False):
        super(ReturnModelNextVit, self).__init__()
        # Initialize the Unet model
        if not model_name.startswith("nextvit"):
            raise ValueError("This Class is only for NextVit models")
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.encoder = timm.create_model(model_name)
        self.decoder = UnetDecoder(
            decoder_channels=self.decoder_channels,
            encoder_channels=(3, 64, 96, 256, 512, 1024),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x)
        features = self.encoder(x)
        for i in features:
            print(i.shape)
        x = self.decoder(*features)
        x = self.segmentation_head(x)
        x = self._unpad(x, original_size, pad)
        return x

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 224):
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


#model = ReturnModelNextVit("nextvit_base", 3, 2)
#model(torch.randn(1, 3, 224, 224))
