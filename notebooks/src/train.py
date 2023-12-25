import os
import warnings
import yaml
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
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss
import bitsandbytes as bnb


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])
    gc.enable()
    accelerate = Accelerator(
        mixed_precision="fp16", log_with=["wandb"],
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=True, ), ],
        project_dir="logs",
        gradient_accumulation_steps=cfg['gradient_accumulation_steps']
    )
    accelerate.init_trackers(project_name="SenNetKidney", config=cfg)
    kidneys_df = pd.read_csv(cfg['kidneys_df'])
    kidney_rle = {kidneys_df['id'][i]: kidneys_df['kidney_rle'][i] for i in range(len(kidneys_df))}
    train_images = os.listdir(f"{cfg['train_dir']}/images/")
    validation_images = os.listdir(f"{cfg['validation_dir']}/images/")
    train_images_xz = os.listdir(f"{cfg['train_dir']}_xz/images/")
    train_images_yz = os.listdir(f"{cfg['train_dir']}_yz/images/")
    train_images_kidney_3 = os.listdir(f"{cfg['train_dir_2']}/images/")
    train_kidneys_rle = list(map(lambda x: kidney_rle[f"kidney_1_dense_{x.split('.')[0]}"], train_images))
    train_images = list(map(lambda x: f"{cfg['train_dir']}/images/{x}", train_images))
    train_masks = list(map(lambda x: x.replace("images", "labels"), train_images))
    validation_kidneys_rle = list(map(lambda x: kidney_rle[f"kidney_3_sparse_{x.split('.')[0]}"], validation_images))
    validation_images = list(map(lambda x: f"{cfg['validation_dir']}/images/{x}", validation_images))
    validation_masks = list(map(lambda x: x.replace("images", "labels"), validation_images))
    train_xz_kidneys_rle = list(map(lambda x: kidney_rle[f"kidney_1_dense_xz_{x.split('.')[0]}"], train_images_xz))
    train_images_xz = list(map(lambda x: f"{cfg['train_dir']}_xz/images/{x}", train_images_xz))
    train_masks_xz = list(map(lambda x: x.replace("images", "labels"), train_images_xz))
    train_yz_kidneys_rle = list(map(lambda x: kidney_rle[f"kidney_1_dense_yz_{x.split('.')[0]}"], train_images_yz))
    train_images_yz = list(map(lambda x: f"{cfg['train_dir']}_yz/images/{x}", train_images_yz))
    train_masks_yz = list(map(lambda x: x.replace("images", "labels"), train_images_yz))

    train_dataset = ImageDataset(train_images, train_masks, get_train_transform(), train_kidneys_rle)
    valid_dataset = ImageDataset(validation_images, validation_masks, get_valid_transform(),
                                 validation_kidneys_rle)
    train_dataset_xz = ImageDataset(train_images_xz, train_masks_xz, get_train_transform(height=928, width=2304),
                                    train_xz_kidneys_rle)
    train_dataset_yz = ImageDataset(train_images_yz, train_masks_yz, get_train_transform(height=2304, width=1312),
                                    train_yz_kidneys_rle)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    train_loader_xz = DataLoader(train_dataset_xz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_yz = DataLoader(train_dataset_yz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True)
    model = ReturnModel(cfg['model_name'], in_channels=cfg['in_channels'], classes=cfg['classes'])
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=float(cfg['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(len(train_loader) * 8),
                                                                     eta_min=float(cfg['min_lr']))
    train_loader, valid_loader, model, optimizer, scheduler, train_loader_yz, train_loader_xz = accelerate.prepare(
        train_loader,
        valid_loader,
        model,
        optimizer, scheduler, train_loader_yz, train_loader_xz
    )

    criterion = SoftBCEWithLogitsLoss()
    best_dice = -1
    validation_df = pd.read_csv(cfg['validation_df'])
    # remove all the rows which do not contain kidney_3_dense in the id column
    validation_df = validation_df[validation_df['id'].str.contains("kidney_3_dense")].reset_index(drop=True)

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
            validation_df=validation_df

        )
        accelerate.wait_for_everyone()
        unwrapped_model = accelerate.unwrap_model(model)
        model_weights = unwrapped_model.state_dict()
        accelerate.save(model_weights, f"{cfg['model_dir']}/model_epoch_{epoch}.pth")
        if dice_score > best_dice:
            best_dice = dice_score
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
