from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_train_transform(DIM) -> Compose:
    return Compose([
        PadIfNeeded(min_height=1312, min_width=928),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        # RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True), ])


def get_train_transform_kidney_2(DIM) -> Compose:
    return Compose([
        PadIfNeeded(min_height=1056, min_width=1536),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        # RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True), ])


def reverse_padding(image, original_height, original_width):
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


def get_valid_transform(DIM) -> Compose:
    return Compose([
        PadIfNeeded(min_height=1728, min_width=1536),
        ToTensorV2(transpose_mask=True),
    ])
