import numpy as np

from utils import *
from torch.utils.data import Dataset
import cv2
from albumentations import Compose
from typing import Tuple, List, Literal
import torch


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform: Compose, kidney_rle: List[str]):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.kidney_rle = kidney_rle
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor,Tuple]:
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        original_shape = image.shape
        mask = cv2.imread(self.mask_paths[item])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / 255
        kidney_mask = rle_decode(self.kidney_rle[item], img_shape=mask.shape)
        mask = np.stack([mask, kidney_mask], axis=2)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask, original_shape


class ImageDatasetOOF(Dataset):
    def __init__(self, image_paths: List[str], transform: Compose, volume: np.array,
                 mode: Literal["xy", "yz", "xz"] = "xy"):
        self.image_paths = image_paths
        self.transform = transform
        self.volume = volume
        self.mode = mode

    def __len__(self) -> int:
        match self.mode:
            case "xy":
                return self.volume.shape[0]
            case "xz":
                return self.volume.shape[1]
            case "yz":
                return self.volume.shape[2]

    def __getitem__(self, item) -> tuple[torch.Tensor, tuple[str, ...], str]:
        if self.mode == "xy":
            image = self.volume[item]
        elif self.mode == "xz":

            image = self.volume[:, item]
        elif self.mode == "yz":
            image = self.volume[:, :, item]
        else:
            raise ValueError("mode must be either xz or yz")

        image_id = self.image_paths[item].split("/")[-1].split(".")[0]
        folder_id = self.image_paths[item].split("/")[-3]
        image_id = f"{folder_id}_{image_id}"
        image_shape = image.shape
        image_shape = tuple(str(element) for element in image_shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        augmented = self.transform(image=image)
        image = augmented["image"]
        return image, image_shape, image_id
