import numpy as np

from utils import *
from torch.utils.data import Dataset
import cv2
from albumentations import Compose
from typing import Tuple, List, Literal
import torch


class ImageDataset(Dataset):
    def __init__(self, transform: Compose, volume: np.array,
                 kidney_volume: np.array, mask_volume: np.array, mode: Literal['xy', 'xz', 'yz'] = "xy", ):
        self.transform = transform
        self.volume = volume
        self.kidney_volume = kidney_volume
        self.mode = mode
        self.mask_volume = mask_volume

    def __len__(self) -> int:
        match self.mode:
            case "xy":
                return self.volume.shape[0]
            case "xz":
                return self.volume.shape[1]
            case "yz":
                return self.volume.shape[2]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        match self.mode:
            case "xy":
                image = self.volume[item]
                kidney_mask = self.kidney_volume[item]
                mask = self.mask_volume[item]
            case "xz":
                image = self.volume[:, item]
                kidney_mask = self.kidney_volume[:, item]
                mask = self.mask_volume[:, item]
            case "yz":
                image = self.volume[:, :, item]
                kidney_mask = self.kidney_volume[:, :, item]
                mask = self.mask_volume[:, :, item]
            case _:
                raise ValueError("mode must be either xz or yz or xy")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)

        mask = mask / 255
        mask = np.stack([mask, kidney_mask], axis=2)
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask


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
