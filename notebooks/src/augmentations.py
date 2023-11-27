from albumentations import *
from albumentations.pytorch import ToTensorV2


def get_train_transform(DIM) -> Compose:
    return Compose([
        RandomResizedCrop(DIM, DIM),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True), ])


def get_test_transform(DIM) -> Compose:
    return Compose([
        Resize(DIM, DIM),
        ToTensorV2(transpose_mask=True),
    ])

