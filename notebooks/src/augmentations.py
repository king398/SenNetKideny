from albumentations import Compose, CenterCrop
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import v2


def get_train_transform(height: int = 1344, width: int = 1120) -> Compose:
    return Compose([
        # RandomBrightnessContrast(p=0.05,),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        ToTensorV2(transpose_mask=True), ])




def CutMix(images: torch.tensor, masks: torch.tensor):
    y = torch.randint(4, (images.shape[0],), dtype=torch.long)
    concat = torch.cat([images, masks], dim=1)
    cutmix = v2.CutMix(num_classes=4)
    cutmixed_images_masks, _ = cutmix(concat, y)
    return cutmixed_images_masks[:, 0:3], cutmixed_images_masks[:, 3:]


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
