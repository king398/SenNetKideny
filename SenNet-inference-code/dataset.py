import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


def reverse_padding(image: np.array, H: int, W: int):
    transform = A.CenterCrop(height=H, width=W)
    return transform(image=image)['image']


def get_valid_transform(image: np.array, H: int, W: int, pad_factor: int = 32) -> np.array:
    h = int(np.ceil(H / pad_factor) * pad_factor)
    w = int(np.ceil(W / pad_factor) * pad_factor)
    transform = A.Compose([
        A.PadIfNeeded(min_height=h, min_width=w),
        ToTensorV2(),
    ])
    return transform(image=image)['image']


def norm_by_percentile(volume, low=10, high=99.8, alpha=0.01):
    q_min = np.percentile(volume, low)
    q_max = np.percentile(volume, high)
    x = (volume - q_min) / (q_max - q_min)
    x[x > 1] = (x[x > 1] - 1) * alpha + 1
    x[x < 0] = (x[x < 0]) * alpha
    return x


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, transform, volume: np.array, pad_factor, mode: str = "xy", ):
        self.image_paths = image_paths
        self.transform = transform
        self.mode = mode
        self.volume = volume
        self.pad_factor = pad_factor

    def __len__(self) -> int:
        match self.mode:
            case "xy":
                return self.volume.shape[0]
            case "xz":
                return self.volume.shape[1]
            case "yz":
                return self.volume.shape[2]

    def __getitem__(self, item):
        match self.mode:
            case "xy":
                image = self.volume[item, :, :]
            case "xz":
                image = self.volume[:, item, :]
            case "yz":
                image = self.volume[:, :, item]
            case _:
                raise ValueError("Invalid mode")

        if self.mode == "xy":
            image_id = self.image_paths[item].split("/")[-1].split(".")[0]
            folder_id = self.image_paths[item].split("/")[-3]
        else:
            image_id = "Na"
            folder_id = "Na"

        image_id = f"{folder_id}_{image_id}"
        image_shape = image.shape
        image_shape = tuple(str(element) for element in image_shape)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transform(image=image, H=int(image_shape[0]), W=int(image_shape[1]), pad_factor=self.pad_factor)
        return image, image_shape, image_id
