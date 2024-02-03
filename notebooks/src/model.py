import segmentation_models_pytorch as smp
import timm
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch.utils.checkpoint import checkpoint
from nextvit import *
from segmentation_models_pytorch.base.heads import SegmentationHead
from torch import nn
import torch
from transformers import UperNetForSemanticSegmentation
from maxvit_decoder import MaxViTDecoder


def return_model(model_name: str, in_channels: int, classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,

    )
    return model


class ReturnModelUperNet(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModelUperNet, self).__init__()
        id2label = {"0": "kidney"}
        label2id = {"kidney": 0}
        self.model = UperNetForSemanticSegmentation.from_pretrained(model_name, id2label=id2label,
                                                                    label2id=label2id,
                                                                    ignore_mismatched_sizes=True)
        self.pad_factor = pad_factor

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_image(x, pad_factor=self.pad_factor)
        x = checkpoint(self.model, x, use_reentrant=True).logits
        # x = self.model.decode_head(*x)
        x = _unpad(x, original_size, pad)
        return x


def _pad_image(x: torch.Tensor, pad_factor: int = 224, pad_value: int = 0):
    h, w = x.shape[2], x.shape[3]
    h_pad = (pad_factor - h % pad_factor) % pad_factor
    w_pad = (pad_factor - w % pad_factor) % pad_factor

    # Calculate padding
    pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
    x = nn.functional.pad(x, pad, mode='replicate')
    return x, pad


def _unpad(x, original_size, pad):
    h, w = original_size
    return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModel, self).__init__()
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )
        self.pad_factor = pad_factor

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = _pad_image(x, pad_factor=self.pad_factor, pad_value=x.min())
        x = checkpoint(self.unet.encoder, x, use_reentrant=True)
        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)
        x = _unpad(x, original_size, pad)
        return x


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

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 32):
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


class ReturnModelMaxViTDecoder(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModelMaxViTDecoder, self).__init__()
        # Initialize the Unet model
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.encoder = timm.create_model(model_name)
        self.decoder = MaxViTDecoder(
            in_channels=(64, 96, 192, 384, 768),
            depths=(2, 2, 2, 2),
            grid_window_size=(7, 7),
            attn_drop=0.2,
            drop=0.2,
            drop_path=0.2,
            debug=True,
            channels=64,
            num_classes=1,
        )
        self.pad_factor = pad_factor

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x, pad_factor=self.pad_factor)
        features = checkpoint(self.encoder, x)
        x = self.decoder(*features)
        x = self._unpad(x, original_size, pad)
        return x

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 32):
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
