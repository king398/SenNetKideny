import gc
from torch.utils.checkpoint import checkpoint
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
import random
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import glob
from torch import nn
import pandas as pd
from typing import Literal
from skimage import filters


def apply_hysteresis_thresholding(volume: np.array, low: float, high: float, chunk_size: int = 2):
    """
    Applies hysteresis thresholding to a 3D numpy array.

    :param volume: 3D numpy array.
    :param low: Low threshold.
    :param high: High threshold.
    :param chunk_size: Size of the chunks to process at once.
    :return: Thresholded volume.
    """
    # Apply hysteresis thresholding to each slice in the volume

    D, H, W = volume.shape
    predict = np.zeros((D, H, W), np.uint8)

    for i in range(0, D, chunk_size // 2):
        predict[i:i + chunk_size] = np.maximum(
            filters.apply_hysteresis_threshold(volume[i:i + chunk_size], low, high),
            predict[i:i + chunk_size]
        )

    return predict


def rle_encode(mask: np.array) -> str:
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle


def get_valid_transform(image: np.array) -> np.array:
    transform = Compose([
        ToTensorV2(),
    ])

    # Apply the transformation
    return transform(image=image)['image']


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class ImageDataset(Dataset):
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
                image = self.volume[item]
            case "xz":
                image = self.volume[:, item]
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
        image = self.transform(image=image, )
        return image, image_shape, image_id


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, pad_factor: int):
        super(ReturnModel, self).__init__()
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )
        self.pad_factor = pad_factor

    def forward(self, x):
        original_size = x.shape[2:]
        x, pad = self._pad_image(x, pad_factor=self.pad_factor)
        x = checkpoint(self.unet.encoder, x, use_reentrant=True)
        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)
        x = self._unpad(x, original_size, pad)
        return x

    def _pad_image(self, x: torch.Tensor, pad_factor: int = 224):
        h, w = x.shape[2], x.shape[3]
        h_pad = (pad_factor - h % pad_factor) % pad_factor
        w_pad = (pad_factor - w % pad_factor) % pad_factor

        # Calculate padding
        pad = [w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2]
        x = nn.functional.pad(x, pad, mode='constant', value=0)
        return x, pad

    def _unpad(self, x, original_size, pad):
        h, w = original_size
        return x[:, :, pad[2]:h + pad[2], pad[0]:w + pad[0]]


def inference_loop(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    gc.collect()
    outputs = None
    counter = 0
    images = (images - images.min()) / (images.max() - images.min() + 0.001)

    with torch.no_grad() and autocast(dtype=torch.float16):
        outputs_batch = model(images).sigmoid().detach().cpu().float()
        outputs = outputs_batch
        counter += 1
        outputs_batch = model(torch.flip(images, dims=[2, ])).sigmoid().detach().cpu().float()
        outputs += torch.flip(outputs_batch, dims=[2, ])
        counter += 1
        outputs_batch = model(torch.flip(images, dims=[3, ])).sigmoid().detach().cpu().float()
        outputs += torch.flip(outputs_batch, dims=[3, ])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=1, dims=[2, 3])).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-1, dims=[2, 3])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=2, dims=[2, 3])).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-2, dims=[2, 3])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=3, dims=[2, 3])).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-3, dims=[2, 3])
        counter += 1

    outputs /= counter
    outputs = outputs.detach().cpu().float()
    return outputs


def inference_fn(model: nn.Module, data_loader: DataLoader, data_loader_xz: DataLoader, data_loader_yz: DataLoader,
                 device: torch.device,
                 volume_shape: Tuple) -> Tuple[list, list]:
    torch.cuda.empty_cache()
    model.eval()
    rles_list = []
    image_ids_all = []
    volume = np.zeros(volume_shape, dtype=np.float16)
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device, non_blocking=True).float()
        outputs: torch.Tensor = inference_loop(model, images)

        for j, image in enumerate(outputs):
            output_mask = image[0, :, :].numpy()
            print(output_mask.shape)
            image_ids_all.append(image_ids[j])
            volume[global_counter] += output_mask
            global_counter += 1
            del image, output_mask
        del outputs, images
        gc.collect()
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_xz), total=len(data_loader_xz)):
        images = images.to(device, non_blocking=True).float()
        outputs = inference_loop(model, images)

        for j, image in enumerate(outputs):
            output_mask = image[0, :, :].numpy()

            volume[:, global_counter] += output_mask
            global_counter += 1
            del image, output_mask
        del outputs, images

    gc.collect()
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_yz), total=len(data_loader_yz)):
        images = images.to(device, non_blocking=True).float()
        outputs = inference_loop(model, images)
        for j, image in enumerate(outputs):
            output_mask = image[0, :, :].numpy()

            volume[:, :, global_counter] += output_mask
            global_counter += 1
            del image, output_mask
        del outputs, images

    gc.collect()
    volume = volume / 3
    volume = apply_hysteresis_thresholding(volume, 0.2, 0.6)

    # volume = volume > 0.2
    # volume = (volume * 255).astype(np.uint8)
    for output_mask in volume:
        # output_mask = output_mask > 0.2
        output_mask = (output_mask * 255).astype(np.uint8)
        rles_list.append(rle_encode(output_mask))
    del volume
    gc.collect()
    return rles_list, image_ids_all


def norm_by_percentile(volume, low=10, high=99.8, alpha=0.01):
    xmin = np.percentile(volume, low)
    xmax = np.percentile(volume, high)
    x = (volume - xmin) / (xmax - xmin)
    if 1:
        x[x > 1] = (x[x > 1] - 1) * alpha + 1
        x[x < 0] = (x[x < 0]) * alpha
    # x = np.clip(x,0,1)
    return x


def main(cfg: dict):
    seed_everything(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dirs = sorted(glob.glob(f"{cfg['test_dir']}/*"))
    model = ReturnModel(cfg['model_name'], cfg['in_channels'], cfg['classes'], pad_factor=224)
    model.to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cuda')), strict=True)
    model = nn.DataParallel(model)

    global_rle_list = []
    global_image_ids = []

    for test_dir in test_dirs:
        test_files = sorted(glob.glob(f"{test_dir}/images/*.tif"))
        print(test_files)
        volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in test_files])
        volume = norm_by_percentile(volume).astype(np.float32)
        print(volume.shape)
        test_dataset_xy = ImageDataset(test_files, get_valid_transform, mode='xy', volume=volume)
        test_dataset_xz = ImageDataset(test_files, get_valid_transform, mode='xz',
                                       volume=volume)
        test_dataset_yz = ImageDataset(test_files, get_valid_transform, mode='yz',
                                       volume=volume)
        test_loader = DataLoader(test_dataset_xy, batch_size=cfg['batch_size'], shuffle=False,
                                 num_workers=cfg['num_workers'], pin_memory=True)
        test_loader_xz = DataLoader(test_dataset_xz, batch_size=cfg['batch_size'], shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=True)
        test_loader_yz = DataLoader(test_dataset_yz, batch_size=cfg['batch_size'], shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=True)
        rles_list, image_ids = inference_fn(model=model, data_loader=test_loader, data_loader_xz=test_loader_xz,
                                            data_loader_yz=test_loader_yz,
                                            device=device, volume_shape=volume.shape[:3])
        global_rle_list.extend(rles_list)
        global_image_ids.extend(image_ids)
        del volume, test_dataset_xy, test_dataset_xz, test_dataset_yz, test_loader, test_loader_xz, test_loader_yz
    submission = pd.DataFrame()
    submission['id'] = global_image_ids
    submission['rle'] = global_rle_list
    submission.to_csv('submission.csv', index=False)
    print(submission.head())


config = {
    "seed": 42,
    "model_name": "tu-timm/maxvit_base_tf_224.in1k",
    "in_channels": 3,
    "classes": 1,
    "test_dir": '/kaggle/input/blood-vessel-segmentation/test',
    "model_path": "/kaggle/input/senet-model-3/maxvit_base_tf_224_fixed_lr_scheduler_no_kidney/model_best_surface_dice.pth",
    "batch_size": 8,
    "num_workers": 4,
    "threshold": 0.15,
}
if __name__ == "__main__":
    main(config)
