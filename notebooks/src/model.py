import timm
import torch
import segmentation_models_pytorch as smp

from nextvit import *
from torch import nn
from torchvision import transforms as T
from torch.utils.checkpoint import checkpoint
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


def return_model(model_name: str, in_channels: int, classes: int):
    return smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes
    )


def central_crop(size):
    return T.CenterCrop(size)


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModel, self).__init__()
        # Initialize the Unet model
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )
        self.pad_factor = pad_factor

    def forward(self, x, qmin, qmax, inference: bool = False):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x, pad_factor=self.pad_factor)
        x = (x - qmin) / (qmax - qmin)
        x[x > 1] = (x[x > 1] - 1) * 0.01 + 1
        x[x < 0] = (x[x < 0]) * 0.01
        x = (x - x.min()) / (x.max() - x.min() + 0.0001)
        x = checkpoint(self.unet.encoder, x, use_reentrant=True)

        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)
        x = self._unpad(x, original_size, pad)
        return x

    @staticmethod
    def _pad_image(x: torch.Tensor, pad_factor: int = 224):
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_factor - h % pad_factor) % pad_factor
        w_pad = (pad_factor - w % pad_factor) % pad_factor
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        return nn.functional.pad(x, pad, mode='constant', value=0), pad

    @staticmethod
    def _unpad(x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]


class ReturnModelDepth6(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, inference: bool = False):
        super(ReturnModelDepth6, self).__init__()
        # Initialize the Unet model
        self.decoder_channels = (2048, 1024, 512, 256, 64, 64)
        self.encoder = timm.create_model(model_name, features_only=True, pretrained=True)
        self.decoder = UnetDecoder(
            decoder_channels=self.decoder_channels,
            encoder_channels=(3, 64, 64, 256, 512, 1024, 2048),
            n_blocks=6,
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
        self.upsample = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.downsample = nn.Conv2d(classes, classes, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x)
        features = []
        xx = checkpoint(self.upsample, x)
        features.append(xx)
        xx = checkpoint(self.encoder.conv1, xx)
        xx = checkpoint(self.encoder.bn1, xx)
        xx = checkpoint(self.encoder.act1, xx)
        features.append(xx)
        del xx
        features.extend(checkpoint(self.encoder, x))
        x = checkpoint(self.decoder, *features)
        x = self.segmentation_head(x)
        x = self.downsample(x)
        x = self._unpad(x, original_size, pad)
        return x

    @staticmethod
    def _pad_image(x: torch.Tensor, pad_factor: int = 32):
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_factor - h % pad_factor) % pad_factor
        w_pad = (pad_factor - w % pad_factor) % pad_factor

        # Calculate padding
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        return x, pad

    @staticmethod
    def _unpad(x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]


class ReturnModelNextVit(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModelNextVit, self).__init__()
        # Initialize the Unet model
        if not model_name.startswith("nextvit"):
            raise ValueError("This Class is only for NextVit models")
        self.decoder_channels = (1024, 512, 256, 96, 64)
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
        self.pad_factor = pad_factor

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x, pad_factor=self.pad_factor)
        features = checkpoint(self.encoder, x)
        x = self.decoder(*features)
        x = self.segmentation_head(x)
        x = self._unpad(x, original_size, pad)
        return x

    @staticmethod
    def _pad_image(x: torch.Tensor, pad_factor: int = 32):
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_factor - h % pad_factor) % pad_factor
        w_pad = (pad_factor - w % pad_factor) % pad_factor
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        return x, pad

    @staticmethod
    def _unpad(x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]
