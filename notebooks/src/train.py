import gc
import os
import yaml
import torch
import argparse
import pandas as pd

from math import ceil
from pathlib import Path
from model import ReturnModel
from dataset import ImageDataset
from torch.utils.data import DataLoader
from train_fn import train_fn, validation_fn
from torch_ema import ExponentialMovingAverage
from segmentation_models_pytorch.losses import DiceLoss

from torch.optim.swa_utils import SWALR, AveragedModel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from accelerate import Accelerator, DistributedDataParallelKwargs
from augmentations import get_fit_transform, get_val_transform
from utils import seed_everything, write_yaml, load_images_and_masks

import warnings

warnings.filterwarnings("ignore")


def main(cfg):
    seed_everything(cfg['seed'])
    gc.enable()
    accelerate = Accelerator(
        mixed_precision="fp16", log_with=["wandb"],
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True,
                                                       find_unused_parameters=True, ), ],
        project_dir="logs",
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
    )
    accelerate.init_trackers(project_name="SenNetKidney", config=cfg)
    fit_images, fit_masks, fit_kidneys_rle, fit_volume = load_images_and_masks(cfg, kidney_name='kidney_1_dense')

    val_images, val_masks, val_kidneys_rle, val_volume = load_images_and_masks(cfg, kidney_name='kidney_3_dense')

    fit_images_xz, fit_masks_xz, fit_xz_kidneys_rle = load_images_and_masks(cfg, kidney_name='kidney_1_dense_xz')
    fit_images_yz, fit_masks_yz, fit_yz_kidneys_rle = load_images_and_masks(cfg, kidney_name='kidney_1_dense_yz')

    # fit_images_3_sparce, fit_masks_3_sparce, fit_kidneys_rle = load_images_and_masks(cfg, kidney_name='kidney_3_sparce')
    # fit_images_3_sparce_xz, fit_masks_3_sparce_xz, fit_xz_kidneys_rle = load_images_and_masks(cfg, kidney_name='kidney_3_sparce_xz')
    # fit_images_3_sparce_yz, fit_masks_3_sparce_yz, fit_yz_kidneys_rle = load_images_and_masks(cfg, kidney_name='kidney_3_sparce_yz')
    #
    # fit_dataset_3 = ImageDataset(fit_images_3_sparce, fit_masks_3_sparce, get_fit_transform(), fit_kidneys_rle, fit_volume, mode="xy")
    # fit_dataset_3_xz = ImageDataset(fit_images_3_sparce_xz, fit_masks_3_sparce_xz, get_fit_transform(), fit_xz_kidneys_rle, fit_volume, mode="xz")
    # fit_dataset_3_yz = ImageDataset(fit_images_3_sparce_yz, fit_masks_3_sparce_yz, get_fit_transform(), fit_yz_kidneys_rle, fit_volume, mode="yz")

    # Load train images and masks for train_dir_2_xz
    fit_dataset = ImageDataset(fit_images, fit_masks, get_fit_transform(), fit_kidneys_rle, fit_volume, mode="xy")
    val_dataset = ImageDataset(val_images, val_masks, get_val_transform(), val_kidneys_rle, val_volume, mode="xy")
    fit_dataset_xz = ImageDataset(fit_images_xz, fit_masks_xz, get_fit_transform(), fit_xz_kidneys_rle, fit_volume, mode="xz")
    fit_dataset_yz = ImageDataset(fit_images_yz, fit_masks_yz, get_fit_transform(), fit_yz_kidneys_rle, fit_volume, mode="yz")

    fit_loader_kwargs = {'batch_size': cfg['batch_size'], 'shuffle': True,
                         'num_workers': cfg['num_workers'], 'pin_memory': True}

    fit_loader = DataLoader(fit_dataset, **fit_loader_kwargs)
    fit_loader_xz = DataLoader(fit_dataset_xz, **fit_loader_kwargs)
    fit_loader_yz = DataLoader(fit_dataset_yz, **fit_loader_kwargs)

    # fit_loader_3 = DataLoader(fit_dataset_3, **fit_loader_kwargs)
    # fit_loader_3_xz = DataLoader(fit_dataset_3_xz, **fit_loader_kwargs)
    # fit_loader_3_yz = DataLoader(fit_dataset_3_yz, **fit_loader_kwargs)

    fit_loader_kwargs['shuffle'] = False
    val_loader = DataLoader(val_dataset, **fit_loader_kwargs)

    model = ReturnModel(cfg['model_name'], in_channels=cfg['in_channels'], classes=cfg['classes'],
                        pad_factor=cfg['pad_factor'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=(float(cfg['lr'])))

    T_max = int((len(fit_loader) + len(fit_loader_yz) + len(fit_loader_xz)
                 # + len(fit_loader_3) + len(fit_loader_3_yz) + len(fit_loader_3_xz)
                 ) * 10)
    # T_max = ceil(len(fit_images + fit_images_xz + fit_images_yz) /
    #              (cfg['num_devices'] * cfg['batch_size'])) * cfg['epochs']
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_max, eta_min=float(cfg['min_lr']))
    swa_start = cfg['swa_start']
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg['swa_lr'], anneal_strategy='linear')

    (fit_loader, val_loader, model, optimizer, scheduler, fit_loader_yz, fit_loader_xz, swa_model, swa_scheduler
     # fit_loader_3, fit_loader_3_yz, fit_loader_3_xz,
     ) = accelerate.prepare(
        fit_loader, val_loader, model, optimizer, scheduler, fit_loader_yz, fit_loader_xz,
        swa_model, swa_scheduler
        # fit_loader_3, fit_loader_3_yz, fit_loader_3_xz
    )

    best_dice = -1
    best_surface_dice = -1
    criterion = DiceLoss(mode="multilabel")
    # remove all the rows which do not contain kidney_3_dense in the id column
    labels_df = pd.read_csv(cfg['labels_df'])
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    for epoch in range(cfg['epochs']):
        train_fn(
            data_loader_list=[fit_loader, fit_loader_xz, fit_loader_yz],
                              # fit_loader_3, fit_loader_3_xz, fit_loader_3_yz],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            ema=ema,
            fold=0,
            accelerator=accelerate,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
            cfg=cfg
        )

        dice_score, surface_dice = validation_fn(
            valid_loader=val_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            accelerator=accelerate,
            labels_df=labels_df,
            swa_model=swa_model,
            cfg=cfg,
            ema=ema,
        )
        accelerate.wait_for_everyone()
        if epoch + 1 > cfg['swa_start']:
            unwrapped_model = accelerate.unwrap_model(swa_model)
            model_weights = unwrapped_model.state_dict()
        else:
            unwrapped_model = accelerate.unwrap_model(model)
            model_weights = unwrapped_model.state_dict()
        if dice_score > best_dice:
            best_dice = dice_score

            accelerate.save(model_weights, f"{cfg['model_dir']}/model.pth")
            accelerate.print(f"Saved Model With Best Dice Score {best_dice}")
        if surface_dice > best_surface_dice:
            best_surface_dice = surface_dice
            accelerate.save(model_weights, f"{cfg['model_dir']}/model_best_surface_dice.pth")
            accelerate.print(f"Saved Model With Best Surface Dice Score {best_surface_dice}")
        if epoch == 3:
            accelerate.save(model_weights, f"{cfg['model_dir']}/model_epoch_{epoch}.pth")
        accelerate.save(model_weights, f"{cfg['model_dir']}/model_last_epoch.pth")
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
