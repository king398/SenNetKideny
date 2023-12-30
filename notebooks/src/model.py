import torch.nn as nn
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.checkpoint import checkpoint
from segmentation_models_pytorch.encoders import TimmUniversalEncoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead


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


class ReturnModelStem(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, inference: bool = False):
        super(ReturnModelStem, self).__init__()
        # Initialize the Unet model
        self.encoder = TimmUniversalEncoder(
            name=model_name,
            in_channels=in_channels,
            depth=4,
            output_stride=32,
            pretrained="imagenet",
        )
        #self.encoder.out_channels.insert(2, 64)
        self.decoder = UnetDecoder(
            decoder_channels=(256, 128, 64, 32, 16),
            encoder_channels=self.encoder.out_channels,
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        # if not inference:
        #    self.unet.encoder.model.set_grad_checkpointing(True)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=4, stride=1)

    def forward(self, x, inference: bool = False):
        # Pad the input
        original_size = x.shape[2:]
        x, pad = self._pad_image(x)
        features = []
        features.append(x)
        xx = self.encoder.model.stem_0(x)
        xx = self.avg_pool(xx)
        xx = self.encoder.model.stem_1(xx)
        features.append(xx)
        x = self.encoder.model.stem_0(x)
        x = self.encoder.model.stem_1(x)
        x = self.encoder.model.stages_0(x)
        features.append(x)
        x = self.encoder.model.stages_1(x)
        features.append(x)
        x = self.encoder.model.stages_2(x)
        features.append(x)
        x = self.encoder.model.stages_3(x)
        features.append(x)
        for i in features:
            print(i.shape)
        x = self.decoder(*features)

        # Forward pass through Unet
        # x = checkpoint(self.unet.encoder, x, use_reentrant=False)

        # x = self.unet.decoder(*x)
        # x = self.unet.segmentation_head(x)
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
