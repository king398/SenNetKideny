import glob
import os
import warnings

import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils import seed_everything, write_yaml
import gc
from dataset import ImageDataset
from pathlib import Path
from augmentations import get_train_transform, get_valid_transform
from torch.utils.data import DataLoader
from model import *
import torch
from train_fn import train_fn, validation_fn
import argparse
from collections import OrderedDict
from sklearn.model_selection import KFold
import cv2
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])
    gc.enable()
    accelerate = Accelerator(
        mixed_precision="fp16", log_with=["tensorboard", ],
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=True), ],
        project_dir="logs"
    )
    accelerate.init_trackers(project_name="SenNetKidney", config=cfg)
    train_images = os.listdir(f"{cfg['train_dir']}/images/")
    train_images = list(map(lambda x: f"{cfg['train_dir']}/images/{x}", train_images))
    train_masks = list(map(lambda x: x.replace("images", "labels"), train_images))
    validation_images = os.listdir(f"{cfg['validation_dir']}/images/")
    validation_images = list(map(lambda x: f"{cfg['validation_dir']}/images/{x}", validation_images))
    validation_masks = list(map(lambda x: x.replace("images", "labels"), validation_images))
    train_dataset = ImageDataset(train_images, train_masks, get_train_transform(cfg['image_size']))
    valid_dataset = ImageDataset(validation_images, validation_masks, get_valid_transform(cfg['image_size']))
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True)
    model = return_model(cfg['model_name'], in_channels=cfg['in_channels'], classes=cfg['classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_loader) * 5),
                                                                     eta_min=float(cfg['min_lr']))
    train_loader, valid_loader, model, optimizer, scheduler = accelerate.prepare(train_loader, valid_loader,
                                                                                 model,
                                                                                 optimizer, scheduler, )

    criterion = SoftBCEWithLogitsLoss()
    best_dice = -1
    for epoch in range(cfg['epochs']):
        train_fn(
            train_loader=train_loader,
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

        )
        accelerate.wait_for_everyone()
        if dice_score > best_dice:
            best_dice = best_dice
            unwrapped_model = accelerate.unwrap_model(model)
            model_weights = unwrapped_model.state_dict()
            accelerate.save(model_weights, f"{cfg['model_dir']}/model.pth")


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