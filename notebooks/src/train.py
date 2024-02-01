import os
import warnings
import yaml
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from utils import seed_everything, write_yaml, load_images_and_masks, load_images_and_masks_pseudo
import gc
from dataset import ImageDataset, ImageDatasetPseudo
from pathlib import Path
from augmentations import get_train_transform, get_valid_transform
from torch.utils.data import DataLoader
from model import ReturnModel
import torch
from train_fn import train_fn, validation_fn
import argparse
from segmentation_models_pytorch.losses import DiceLoss


def main(cfg):
    warnings.filterwarnings("ignore")
    seed_everything(cfg['seed'])
    gc.enable()
    accelerate = Accelerator(
        mixed_precision="fp16", log_with=["wandb"],
        kwargs_handlers=[DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=True, ), ],
        project_dir="logs",
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
    )
    accelerate.init_trackers(project_name="SenNetKidney", config=cfg)
    kidneys_df = pd.read_csv(cfg['kidneys_df'])
    kidney_rle = {kidneys_df['id'][i]: kidneys_df['kidney_rle'][i] for i in range(len(kidneys_df))}
    train_images, train_masks, train_kidneys_rle, train_volume = load_images_and_masks(
        cfg['train_dir'], 'images', 'labels', kidney_rle, 'kidney_1_dense'
    )

    # Load validation images and masks
    validation_images, validation_masks, validation_kidneys_rle, validation_volume = load_images_and_masks(
        cfg['validation_dir'], 'images', 'labels', kidney_rle, 'kidney_3_dense'
    )

    # Load train images and masks for train_dir_2

    # Load train images and masks for train_dir_xz
    train_images_xz, train_masks_xz, train_xz_kidneys_rle = load_images_and_masks(
        cfg['train_dir'] + '_xz', 'images', 'labels', kidney_rle, 'kidney_1_dense_xz'
    )

    # Load train images and masks for train_dir_yz
    train_images_yz, train_masks_yz, train_yz_kidneys_rle = load_images_and_masks(
        cfg['train_dir'] + '_yz', 'images', 'labels', kidney_rle, 'kidney_1_dense_yz'
    )

    train_dataset = ImageDataset(train_images, train_masks, get_train_transform(), train_kidneys_rle, train_volume,
                                 mode="xy")
    valid_dataset = ImageDataset(validation_images, validation_masks, get_valid_transform(),
                                 validation_kidneys_rle, validation_volume, mode="xy")
    train_dataset_xz = ImageDataset(train_images_xz, train_masks_xz, get_train_transform(),
                                    train_xz_kidneys_rle, train_volume, mode="xz")
    train_dataset_yz = ImageDataset(train_images_yz, train_masks_yz, get_train_transform(),
                                    train_yz_kidneys_rle, train_volume, mode="yz")
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    train_loader_xz = DataLoader(train_dataset_xz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    train_loader_yz = DataLoader(train_dataset_yz, batch_size=cfg['batch_size'], shuffle=True,
                                 num_workers=cfg['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True)

    model = ReturnModel(cfg['model_name'], in_channels=cfg['in_channels'], classes=cfg['classes'],
                        pad_factor=cfg['pad_factor'], )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(
        (len(train_loader) + len(train_loader_yz) + len(train_loader_xz)) * 10),
                                                                     eta_min=float(cfg['min_lr']))
    (train_loader, valid_loader, model, optimizer, scheduler, train_loader_yz, train_loader_xz,
     ) = accelerate.prepare(
        train_loader,
        valid_loader,
        model,
        optimizer, scheduler, train_loader_yz, train_loader_xz,
    )

    criterion = DiceLoss(mode="multilabel")
    best_dice = -1
    best_surface_dice = -1
    # remove all the rows which do not contain kidney_3_dense in the id column
    labels_df = pd.read_csv(cfg['labels_df'])

    for epoch in range(cfg['epochs']):
        train_fn(
            data_loader_list=[train_loader, train_loader_xz, train_loader_yz],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            fold=0,
            accelerator=accelerate,

        )

        dice_score, surface_dice = validation_fn(
            valid_loader=valid_loader,
            model=model,
            criterion=criterion,
            epoch=epoch,
            accelerator=accelerate,
            labels_df=labels_df,
            model_dir=cfg['model_dir'],

        )
        accelerate.wait_for_everyone()
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
        # accelerate.save(model_weights, f"{cfg['model_dir']}/model_last_epoch.pth")
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
