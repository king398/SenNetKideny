import glob
from torch.utils.data import Dataset
import cv2
from albumentations import Compose
from typing import Tuple
import torch


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform: Compose):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        mask = cv2.imread(self.mask_paths[item])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / 255
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"][:, :, None]
        return image, mask


