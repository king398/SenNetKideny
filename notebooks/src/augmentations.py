from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_train_transform(DIM) -> Compose:
    return Compose([
        PadIfNeeded(min_height=1312, min_width=928),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        #RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True), ])


def get_valid_transform(DIM) -> Compose:
    return Compose([
        PadIfNeeded(min_height=1728, min_width=1536),
        ToTensorV2(transpose_mask=True),
    ])

