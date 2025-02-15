import numpy as np
from augmentations import random_scale
from utils import rle_decode
from torch.utils.data import Dataset
import cv2
from albumentations import Compose
from typing import Tuple, List, Literal
import torch
import random


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform: Compose, kidney_rle: List[str],
                 volume: np.array, mode: Literal["xy", "yz", "xz"] = "xy", train: bool = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.kidney_rle = kidney_rle
        self.transform = transform
        self.volume = volume
        self.mode = mode
        self.train = train

    def __len__(self) -> int:
        match self.mode:
            case "xy":
                return self.volume.shape[0]
            case "xz":
                return self.volume.shape[1]
            case "yz":
                return self.volume.shape[2]

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, str]:
        match self.mode:
            case "xy":
                image = self.volume[item].astype(np.float32)
            case "xz":
                image = self.volume[:, item].astype(np.float32)
            case "yz":
                image = self.volume[:, :, item].astype(np.float32)
            case _:
                raise ValueError("Invalid mode")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.mask_paths[item])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / 255
        kidney_mask = rle_decode(self.kidney_rle[item], img_shape=mask.shape)
        mask = np.stack([mask, kidney_mask], axis=2)

        augmented = self.transform(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]
        image_id = self.image_paths[item].split("/")[-1].split(".")[0]
        folder_id = self.image_paths[item].split("/")[-3]
        image_id = f"{folder_id}_{image_id}"
        return image, mask, image_id


class ImageDatasetPseudo(Dataset):
    def __init__(self, image_paths: List[str], transform: Compose, mask_volume: np.array,
                 volume: np.array, mode: Literal["xy", "yz", "xz"] = "xy", ):
        self.image_paths = image_paths
        self.transform = transform
        self.volume = volume
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

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, str]:
        match self.mode:
            case "xy":
                image = self.volume[item].astype(np.float32)
                mask = self.mask_volume[item].astype(np.float32)
            case "xz":
                image = self.volume[:, item].astype(np.float32)
                mask = self.mask_volume[:, item].astype(np.float32)
            case "yz":
                image = self.volume[:, :, item].astype(np.float32)
                mask = self.mask_volume[:, :, item].astype(np.float32)
            case _:
                raise ValueError("Invalid mode")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        image_id = self.image_paths[item].split("/")[-1].split(".")[0]
        folder_id = self.image_paths[item].split("/")[-3]
        image_id = f"{folder_id}_{image_id}"
        return image, mask, image_id


class ImageDatasetPseudo(Dataset):
    def __init__(self, image_paths: List[str], transform: Compose, mask_volume: np.array,
                 volume: np.array, mode: Literal["xy", "yz", "xz"] = "xy", ):
        self.image_paths = image_paths
        self.transform = transform
        self.volume = volume
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

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, str]:
        match self.mode:
            case "xy":
                image = self.volume[item].astype(np.float32)
                mask = self.mask_volume[item].astype(np.float32)
            case "xz":
                image = self.volume[:, item].astype(np.float32)
                mask = self.mask_volume[:, item].astype(np.float32)
            case "yz":
                image = self.volume[:, :, item].astype(np.float32)
                mask = self.mask_volume[:, :, item].astype(np.float32)
            case _:
                raise ValueError("Invalid mode")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        image_id = self.image_paths[item].split("/")[-1].split(".")[0]
        folder_id = self.image_paths[item].split("/")[-3]
        image_id = f"{folder_id}_{image_id}"
        return image, mask, image_id


class ImageDatasetOOF(Dataset):
    def __init__(self, image_paths: list, transform, volume: np.array,
                 mode: Literal["xy", "yz", "xz"] = "xy", ):

        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode
        self.volume = volume

    def __len__(self) -> int:
        match self.mode:
            case "xy":
                return self.volume.shape[0]
            case "xz":
                return self.volume.shape[1]
            case "yz":
                return self.volume.shape[2]

    def __getitem__(self, item) -> tuple[torch.Tensor, Tuple, str]:
        match self.mode:
            case "xy":
                image = self.volume[item].astype(np.float32)
            case "xz":
                image = self.volume[:, item].astype(np.float32)
            case "yz":
                image = self.volume[:, :, item].astype(np.float32)
            case _:
                raise ValueError("Invalid mode")

        if self.mode == "xy":
            image_id = self.image_paths[item].split("/")[-1].split(".")[0]
            folder_id = self.image_paths[item].split("/")[-3]
        else:
            image_id = "not_applicable"
            folder_id = "not_applicable"
        image_id = f"{folder_id}_{image_id}"
        image_shape = image.shape
        image_shape = tuple(str(element) for element in image_shape)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = image.astype("float")
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        image = self.transform(image=image)
        return image, image_shape, image_id


class CombinedDataLoader:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        return self

    def __next__(self):
        # Randomly pick a dataloader
        dataloader = random.choice(self.iterators)

        # Try to fetch the next batch from this dataloader
        try:
            batch = next(dataloader)
        except StopIteration:
            # If this dataloader is exhausted, remove it from the list
            self.iterators.remove(dataloader)
            if not self.iterators:
                raise StopIteration
            return self.__next__()

        return batch

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloaders)
