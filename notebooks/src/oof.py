import os

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import gc
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import glob
import pandas as pd
from utils import *
from model import ReturnModel
from dataset import ImageDatasetOOF


def get_valid_transform(image: np.array, ) -> np.array:
    """

    :param image: Padded image.
    :return: Cropped image with original dimensions.
    """
    transform = Compose([
        ToTensorV2(),
    ])

    # Apply the transformation
    return transform(image=image)['image']


def inference_loop(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    gc.collect()
    outputs = None
    counter = 0
    with torch.no_grad() and autocast():
        outputs_batch = model(images, ).sigmoid().detach().cpu().float()
        outputs = outputs_batch
        counter += 1
        outputs_batch = model(torch.flip(images, dims=[2, ]), ).sigmoid().detach().cpu().float()
        outputs += torch.flip(outputs_batch, dims=[2, ])
        counter += 1
        outputs_batch = model(torch.flip(images, dims=[3, ]), ).sigmoid().detach().cpu().float()
        outputs += torch.flip(outputs_batch, dims=[3, ])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=1, dims=[2, 3]), ).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-1, dims=[2, 3])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=2, dims=[2, 3]), ).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-2, dims=[2, 3])
        counter += 1
        outputs_batch = model(torch.rot90(images, k=3, dims=[2, 3]), ).sigmoid().detach().cpu().float()
        outputs += torch.rot90(outputs_batch, k=-3, dims=[2, 3])
        counter += 1

    outputs /= counter
    outputs = outputs.detach().cpu().float()
    return outputs


def inference_fn(model: nn.Module, data_loader: DataLoader, data_loader_xz: DataLoader, data_loader_yz: DataLoader,
                 device: torch.device,
                 volume_shape: Tuple) -> Tuple[list, list, np.array]:
    torch.cuda.empty_cache()
    model.eval()
    rles_list = []
    image_ids_all = []
    volume = torch.zeros(volume_shape, dtype=torch.float16)
    global_counter = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device, non_blocking=True).float()
        outputs: torch.Tensor = inference_loop(model, images)

        for j, image in enumerate(outputs):
            output_mask = image[0, :, :]

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
            output_mask = image[0, :, :]

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
            output_mask = image[0, :, :]

            volume[:, :, global_counter] += output_mask
            global_counter += 1
            del image, output_mask
        del outputs, images

    gc.collect()
    volume = volume / 3
    volume = volume.numpy()
    volume_no_threshold = volume.copy()
    volume = apply_hysteresis_thresholding(volume, 0.2, 0.6)
    # volume = volume > 0.3
    volume = (volume * 255).astype(np.uint8)
    for output_mask in volume:
        rles_list.append(rle_encode(output_mask))
    del volume
    gc.collect()
    return rles_list, image_ids_all, volume_no_threshold


def main(cfg: dict):
    global volume_uncompressed
    seed_everything(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dirs = ["/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_3_sparse", ]
    model = ReturnModel(cfg['model_name'], cfg['in_channels'], cfg['classes'], cfg['pad_factor'])
    model.to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cuda')))
    valid_rle = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_3_dense_full.csv")
    valid_rle['id'] = valid_rle['id'].apply(lambda x: x.replace("kidney_3_dense", "kidney_3_sparse"))
    global_rle_list = []
    global_image_ids = []

    for test_dir in test_dirs:
        test_files = sorted(glob.glob(f"{test_dir}/images/*.tif"))

        volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in test_files])
        volume = norm_by_percentile(volume).astype(np.float32)
        test_dataset_xy = ImageDatasetOOF(test_files, get_valid_transform, mode='xy', volume=volume)
        test_dataset_xz = ImageDatasetOOF(test_files, get_valid_transform, mode='xz',
                                          volume=volume)
        test_dataset_yz = ImageDatasetOOF(test_files, get_valid_transform, mode='yz',
                                          volume=volume)
        test_loader = DataLoader(test_dataset_xy, batch_size=cfg['batch_size'] * 4, shuffle=False,
                                 num_workers=cfg['num_workers'], pin_memory=True)
        test_loader_xz = DataLoader(test_dataset_xz, batch_size=cfg['batch_size'] * 2, shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=True)
        test_loader_yz = DataLoader(test_dataset_yz, batch_size=cfg['batch_size'] * 2, shuffle=False,
                                    num_workers=cfg['num_workers'], pin_memory=True)
        rles_list, image_ids, volume_uncompressed = inference_fn(model=model, data_loader=test_loader,
                                                                 data_loader_xz=test_loader_xz,
                                                                 data_loader_yz=test_loader_yz,
                                                                 device=device, volume_shape=volume.shape[:3])
        global_rle_list.extend(rles_list)
        global_image_ids.extend(image_ids)

        del volume, test_dataset_xy, test_dataset_xz, test_dataset_yz, test_loader, test_loader_xz, test_loader_yz

    submission = pd.DataFrame()
    submission['id'] = global_image_ids
    submission['rle'] = global_rle_list
    # get dir path from model path
    model_dir = os.path.dirname(cfg["model_path"])
    submission.to_csv(f"{model_dir}/oof_csv.csv", index=False)
    np.savez_compressed(f"{model_dir}/oof_volume.npz", volume=volume_uncompressed)
    print(submission.head())


config = {
    "seed": 42,
    "model_name": "tu-timm/dm_nfnet_f0.dm_in1k",
    "in_channels": 3,
    "classes": 2,
    # "test_dir": '/kaggle/input/blood-vessel-segmentation/test',
    "model_path": "/home/mithil/PycharmProjects/SenNetKideny/models/dm_nfnet_f0_volume_normalize/model_best_surface_dice.pth",
    "batch_size": 4,
    "num_workers": 8,
    "pad_factor": 32,
}
if __name__ == "__main__":
    main(config)
