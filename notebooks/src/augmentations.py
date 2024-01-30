import cv2
import numpy as np
from albumentations import Compose, CenterCrop, RandomScale, PadIfNeeded, Affine
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import v2
from torchvision.transforms import transforms as T
from typing import Tuple


def get_train_transform(height: int = 1344, width: int = 1120) -> Compose:
    return Compose([
        # RandomBrightnessContrast(p=0.05,),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        Affine(p=0.2, shear=10, rotate=10, scale=0.1, interpolation=cv2.INTER_CUBIC),
        ToTensorV2(transpose_mask=True), ])


def central_crop(size):
    return T.CenterCrop(size)


def get_mosaic_2x2(four_images_batch):
    H, W = four_images_batch.shape[2:]
    h, w = int(H // 2), int(W / 2)
    cc = central_crop((h, w))
    final = torch.zeros_like(four_images_batch[0])
    for i in range(2):
        for j in range(2):
            im = cc(four_images_batch[i * 2 + j])
            final[:, i * h:i * h + h, j * w:j * w + w] = im
    return torch.cat([four_images_batch, final.unsqueeze(0)])


def CutMix(images: torch.tensor, masks: torch.tensor):
    y = torch.randint(4, (images.shape[0],), dtype=torch.long)
    concat = torch.cat([images, masks], dim=1)
    cutmix = v2.CutMix(num_classes=4)
    cutmixed_images_masks, _ = cutmix(concat, y)
    return cutmixed_images_masks[:, 0:3], cutmixed_images_masks[:, 3:]


def random_scale(image: np.array, mask: np.array, original_shape: Tuple[int, int]):
    height, width = original_shape
    transform = Compose([
        RandomScale(scale_limit=0.2, p=1.0),  # scale_limit can be adjusted
        PadIfNeeded(min_height=height, min_width=width, p=1.0),
        CenterCrop(height=height, width=width, p=1.0)
    ])
    transform_image = transform(image=image, mask=mask)
    return transform_image['image'], transform_image['mask']


def reverse_padding(image: int, original_height: int, original_width: int):
    """
    Crops the padded image back to its original dimensions.

    :param image: Padded image.
    :param original_height: Original height of the image before padding.
    :param original_width: Original width of the image before padding.
    :return: Cropped image with original dimensions.
    """
    # Define the cropping transformation
    transform = CenterCrop(height=original_height, width=original_width)

    # Apply the transformation
    return transform(image=image)['image']


def get_valid_transform() -> Compose:
    return Compose([
        ToTensorV2(transpose_mask=True),
    ])
