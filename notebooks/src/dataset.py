import cv2
import torch
import random
import numpy as np

from albumentations import Compose
from torch.utils.data import Dataset
from typing import Tuple, List, Literal


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str],
                 transform: Compose, kidney_rle: List[str],
                 volume: np.array, mode: Literal["xy", "yz", "xz"] = "xy", ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.kidney_rle = kidney_rle
        self.transform = transform
        self.volume = volume
        self.mode = mode
        print('dataset is initialized')

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
                image = self.volume[:, item].astype(np.float32).transpose(1, 0)
            case "yz":
                image = self.volume[:, :, item].astype(np.float32).transpose(1, 0)
            case _:
                raise ValueError("Invalid mode")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        mask = cv2.imread(self.mask_paths[item], cv2.COLOR_BGR2GRAY)
        kidney_mask = cv2.imread(self.kidney_rle[item], cv2.COLOR_BGR2GRAY)
        print(mask.shape, image.shape)

        augmented = self.transform(image=image, masks=[mask, kidney_mask])
        image = augmented["image"]
        mask = np.dstack(augmented["masks"])/255
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
        # image = (image - image.min()) / (image.max() - image.min() + 0.0001)
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


if __name__ == '__main__':
    from utils import load_images_and_masks
    import yaml
    from augmentations import get_fit_transform

    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    fit_images, fit_masks, fit_kidneys_rle, fit_volume = load_images_and_masks(cfg, kidney_name='kidney_1_dense')
    fit_dataset = ImageDataset(fit_images, fit_masks, get_fit_transform(), fit_kidneys_rle, fit_volume, mode="xy")
    print()
