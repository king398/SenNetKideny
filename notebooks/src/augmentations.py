import torch

from torchvision.transforms import v2
from torchvision.transforms import transforms as T
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, CenterCrop


def get_fit_transform():
    return Compose([
        # RandomBrightnessContrast(p=0.05,),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.5),
        ToTensorV2(transpose_mask=True), ])


def get_val_transform() -> Compose:
    return Compose([
        ToTensorV2(transpose_mask=True),
    ])


def central_crop(size):
    return T.CenterCrop(size)


def get_mosaic_2x2(four_images_batch):
    H, W = four_images_batch.shape[2:]
    h, w = int(H // 2), int(W / 2)
    cc = central_crop((h, w))
    final = torch.zeros_like(four_images_batch[0])
    for i in range(2):
        for j in range(2):
            im = cc(four_images_batch[i * 2 + j])   # takes a center crop of each image
            final[:, i * h:i * h + h, j * w:j * w + w] = im
    return torch.cat([four_images_batch, final.unsqueeze(0)])


def CutMix(images: torch.tensor, masks: torch.tensor):
    y = torch.randint(4, (images.shape[0],), dtype=torch.long)
    concat = torch.cat([images, masks], dim=1)
    cutmix = v2.CutMix(num_classes=4)
    cutmixed_images_masks, _ = cutmix(concat, y)
    return cutmixed_images_masks[:, 0:3], cutmixed_images_masks[:, 3:]


def reverse_padding(image: int, original_height: int, original_width: int):
    transform = CenterCrop(height=original_height, width=original_width)
    return transform(image=image)['image']
