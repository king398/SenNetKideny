from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_train_transform(height: int = 1344, width: int = 1120) -> Compose:
    return Compose([
        # RandomBrightnessContrast(p=0.05,),
        ToTensorV2(transpose_mask=True), ])


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
