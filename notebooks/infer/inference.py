import gc
from torch.utils.checkpoint import checkpoint
import re
import albumentations
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
from albumentations import CenterCrop
from typing import Literal
from skimage import filters


def apply_hysteresis_thresholding(volume: np.array, low: float, high: float, chunk_size: int = 32):
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


def choose_biggest_object(mask: np.array, threshold: float) -> np.array:
    mask = ((mask > threshold) * 255).astype(np.uint8)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_label = -1
    max_area = -1
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= max_area:
            max_area = stats[l, cv2.CC_STAT_AREA]
            max_label = l
    processed = (label == max_label).astype(np.uint8)
    return processed


def rle_encode(mask: np.array) -> str:
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle


def get_valid_transform(image: np.array, original_height: int, original_width: int,pad_factor:int=32) -> np.array:
    """
    Crops the padded image back to its original dimensions.

    :param image: Padded image.
    :param original_height: Original height of the image before padding.
    :param original_width: Original width of the image before padding.
    :return: Cropped image with original dimensions.
    """
    # Define the cropping transformation
    # round up original height and width to nearest pad_factor
    original_height = int(np.ceil(original_height / pad_factor) * pad_factor)
    original_width = int(np.ceil(original_width / pad_factor) * pad_factor)
    transform = Compose([
        PadIfNeeded(min_height=original_height, min_width=original_width),
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
            image_id = "Na"
            folder_id = "Na"
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_id = f"{folder_id}_{image_id}"
        image_shape = image.shape
        image_shape = tuple(str(element) for element in image_shape)

        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        image = self.transform(image=image, original_height=int(image_shape[0]), original_width=int(image_shape[1]))
        return image, image_shape, image_id


class ReturnModel(nn.Module):
    def __init__(self, model_name: str, in_channels: int, classes: int, ):
        super(ReturnModel, self).__init__()
        # Initialize the Unet model
        self.unet = smp.Unet(
            encoder_name=model_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )
        # if not inference:
        #    self.unet.encoder.model.set_grad_checkpointing(True)

    def forward(self, x, ):
        # Pad the input
        x = checkpoint(self.unet.encoder, x, )
        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)

        return x


def reverse_padding(image: np.array, original_height: int, original_width: int):
    """
    Crops the padded image back to its original dimensions.

    :param image: Padded image.
    :param original_height: Original height of the image before padding.
    :param original_width: Original width of the image before padding.
    :return: Cropped image with original dimensions.
    """
    # Define the cropping transformation
    transform = CenterCrop(height=original_height, width=original_width)

    # Apply the transformation
    return transform(image=image)['image']


def inference_loop(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    model.eval()
    gc.collect()
    outputs = None
    counter = 0
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
    volume = np.zeros(volume_shape,dtype=np.float16)
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device, non_blocking=True).float()

        outputs = inference_loop(model, images)

        for j, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :]
            output_mask = output_mask.numpy() * kidney

            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][j]),
                                          original_width=int(image_shapes[1][j]))
            image_ids_all.append(image_ids[j])
            volume[global_counter] += output_mask
            global_counter += 1
            del image, output_mask, kidney
        del outputs, images
        gc.collect()
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_xz), total=len(data_loader_xz)):
        images = images.to(device, non_blocking=True).float()
        outputs = inference_loop(model, images)

        for j, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :]
            output_mask = (output_mask.numpy() * kidney)
            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][j]),
                                          original_width=int(image_shapes[1][j]))

            volume[:, global_counter] += output_mask
            global_counter += 1
            del image, output_mask, kidney
        del outputs, images

    gc.collect()
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_yz), total=len(data_loader_yz)):
        images = images.to(device, non_blocking=True).float()
        outputs = inference_loop(model, images)
        for j, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :] * kidney
            output_mask = (output_mask.numpy() * kidney)
            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][j]),
                                          original_width=int(image_shapes[1][j]))

            volume[:, :, global_counter] += output_mask
            global_counter += 1
            del output_mask, image, kidney
        del outputs, images

    gc.collect()
    volume = volume / 3
    #volume = apply_hysteresis_thresholding(volume, 0.2, 0.6)

    volume = volume > 0.2
    volume = (volume * 255).astype(np.uint8)
    for output_mask in volume:
        rles_list.append(rle_encode(output_mask))
    del volume
    gc.collect()
    return rles_list, image_ids_all

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else None
def norm_by_percentile(volume, low=10, high=99.8, alpha=0.01):
    xmin = np.percentile(volume,low)
    xmax = np.percentile(volume,high)
    x = (volume-xmin)/(xmax-xmin)
    if 1:
        x[x>1]=(x[x>1]-1)*alpha +1
        x[x<0]=(x[x<0])*alpha
    #x = np.clip(x,0,1)
    return x

def main(cfg: dict):
    seed_everything(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dirs = sorted(glob.glob(f"{cfg['test_dir']}/*"))
    model = ReturnModel(cfg['model_name'], cfg['in_channels'], cfg['classes'])
    model.to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cuda')), strict=True)
    model = nn.DataParallel(model)

    global_rle_list = []
    global_image_ids = []

    for test_dir in test_dirs:
        test_files = sorted(glob.glob(f"{test_dir}/images/*.tif"))
        print(test_files)
        volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE).astype(np.float16) for i in test_files])
        volume = norm_by_percentile(volume)
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
    "model_name": "tu-timm/seresnext50_32x4d.gluon_in1k",
    "in_channels": 3,
    "classes": 2,
    "test_dir": '/kaggle/input/blood-vessel-segmentation/test',
    "model_path": "/kaggle/input/senet-models/seresnext50_multiview_30_epoch_5e_04_dice_loss_normalize_hflip_3_channels.csv/model.pth",
    "batch_size": 4,
    "num_workers": 0,
    "threshold": 0.15,
}
if __name__ == "__main__":
    main(config)
