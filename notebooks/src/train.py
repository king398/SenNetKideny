import glob
import os
import warnings

import numpy as np
import yaml
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils import seed_everything, write_yaml, rle_decode
import gc
from dataset import ImageDataset
from pathlib import Path
from augmentations import get_train_transform, get_valid_transform
from torch.utils.data import DataLoader
from model import *
import torch
from train_fn import train_fn, validation_fn
import argparse
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
import bitsandbytes as bnb
import cv2
import glob


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])
    gc.enable()
    accelerate = Accelerator(
        mixed_precision="fp16", log_with=["wandb"],
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=False), ],
        project_dir="logs"
    )
    accelerate.init_trackers(project_name="SenNetKidney", config=cfg)
    kidneys_df = pd.read_csv(cfg['kidneys_df'])
    kidney_rle = {kidneys_df['id'][i]: kidneys_df['kidney_rle'][i] for i in range(len(kidneys_df))}
    train_images = glob.glob(f"{cfg['train_dir']}/images/*.tif")
    validation_images = glob.glob(f"{cfg['validation_dir']}/images/*.tif")
    train_kidneys_rle = list(
        map(lambda x: kidney_rle[f"kidney_1_dense_{x.split('.')[0]}"], os.listdir(f"{cfg['train_dir']}/images")))
    train_masks = list(map(lambda x: x.replace("images", "labels"), train_images))
    validation_kidneys_rle = list(
        map(lambda x: kidney_rle[f"kidney_3_sparse_{x.split('.')[0]}"], os.listdir(f"{cfg['validation_dir']}/images")))
    validation_masks = list(map(lambda x: x.replace("images", "labels"), validation_images))
    train_volume = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in train_images])
    print(train_volume.shape)
    train_kidney_volume = np.stack([rle_decode(f, (1303, 912)) for f in train_kidneys_rle])
    train_mask_volume = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in train_masks])
    validation_volume = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in validation_images])
    validation_kidney_volume = np.stack([rle_decode(f, (1706, 1510)) for f in validation_kidneys_rle])
    validation_mask_volume = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in validation_masks])

    train_dataset = ImageDataset(transform=get_train_transform(), volume=train_volume,
                                 kidney_volume=train_kidney_volume, mask_volume=train_mask_volume, mode="xy")
    valid_dataset = ImageDataset(transform=get_valid_transform(), volume=validation_volume,
                                 kidney_volume=validation_kidney_volume, mask_volume=validation_mask_volume, mode="xy")
    train_dataset_xz = ImageDataset(transform=get_train_transform(height=2464, width=1120), volume=train_volume,
                                    kidney_volume=train_kidney_volume, mask_volume=train_mask_volume, mode="xz")
    train_dataset_yz = ImageDataset(get_train_transform(height=2464, width=1344), volume=train_volume,
                                    kidney_volume=train_kidney_volume, mask_volume=train_mask_volume, mode="yz")
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_xz = DataLoader(train_dataset_xz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_yz = DataLoader(train_dataset_yz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    model = ReturnModel(cfg['model_name'], in_channels=cfg['in_channels'], classes=cfg['classes'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_loader) * 5),
                                                                     eta_min=float(cfg['min_lr']))
    train_loader, valid_loader, model, optimizer, scheduler, train_loader_yz, train_loader_xz = accelerate.prepare(
        train_loader,
        valid_loader,
        model,
        optimizer, scheduler, train_loader_yz, train_loader_xz
    )
    valid_rle_df = None
    criterion = SoftBCEWithLogitsLoss()
    best_dice = -1
    for epoch in range(cfg['epochs']):
        train_fn(

            train_loader=train_loader,
            train_loader_xz=train_loader_xz,
            train_loader_yz=train_loader_yz,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            fold=0,
            accelerator=accelerate,

        )

        dice_score = validation_fn(
            valid_loader=valid_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            fold=0,
            accelerator=accelerate,
            validation_df=valid_rle_df

        )
        accelerate.wait_for_everyone()

        unwrapped_model = accelerate.unwrap_model(model)
        model_weights = unwrapped_model.state_dict()
        accelerate.save(model_weights, f"{cfg['model_dir']}/model_epoch_{epoch}.pth")

    accelerate.end_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    os.makedirs(cfg['model_dir'], exist_ok=True)
    write_yaml(cfg, save_path=f"{cfg['model_dir']}/config.yaml")
    main(cfg)
