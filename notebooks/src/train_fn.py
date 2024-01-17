import torch
from tqdm.auto import tqdm
import gc
from torch.utils.data import DataLoader
from utils import *
from torch.nn import Module
from torch import optim
from accelerate import Accelerator
from torch.cuda.amp import autocast
import torch.nn.functional as F
from augmentations import reverse_padding

dice = Dice()
dice_valid = Dice_Valid()


def get_color_escape(r, g, b, background=False):
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'


tqdm_color = get_color_escape(255, 0, 0)  # Red color for example
tqdm_style = {
    'bar_format': f'{tqdm_color}{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]{get_color_escape(255, 255, 255)}'}


def train_fn(
        train_loader: DataLoader,
        train_loader_xz: DataLoader,
        train_loader_yz: DataLoader,
        model: Module,
        criterion: Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        epoch: int,
        accelerator: Accelerator,
        fold: int,

):
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    loss_metric = 0

    stream = tqdm(train_loader, total=len(train_loader), disable=not accelerator.is_local_main_process, **tqdm_style)
    for i, (images, masks) in enumerate(stream):
        masks = masks.float()
        images = images.float()
        output = model(images)
        loss = criterion(output, masks)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        outputs, masks = accelerator.gather_for_metrics((output, masks))
        loss_metric += loss.item() / (i + 1)
        dice_batch = dice(outputs, masks)
        stream.set_description(
            f"Epoch:{epoch + 1}, train_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
        scheduler.step()
        accelerator.log({f"train_loss_{fold}": loss_metric, f"train_dice_batch_{fold}": dice_batch.item(),
                         f"lr_{fold}": optimizer.param_groups[0]['lr']})
    stream_xz = tqdm(train_loader_xz, total=len(train_loader_xz), disable=not accelerator.is_local_main_process,
                     **tqdm_style)

    for i, (images, masks) in enumerate(stream_xz):
        masks = masks.float()
        images = images.float()
        output = model(images)
        loss = criterion(output, masks)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        outputs, masks = accelerator.gather_for_metrics((output, masks))
        loss_metric += loss.item() / (i + 1)
        dice_batch = dice(outputs, masks)
        stream_xz.set_description(
            f"Epoch:{epoch + 1}, train_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
        scheduler.step()
        accelerator.log({f"train_loss_{fold}": loss_metric, f"train_dice_batch_{fold}": dice_batch.item(),
                         f"lr_{fold}": optimizer.param_groups[0]['lr']})
    torch.cuda.empty_cache()
    stream_yz = tqdm(train_loader_yz, total=len(train_loader_yz), disable=not accelerator.is_local_main_process,
                     **tqdm_style)

    for i, (images, masks) in enumerate(stream_yz):
        masks = masks.float()
        images = images.float()
        output = model(images)
        loss = criterion(output, masks)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        outputs, masks = accelerator.gather_for_metrics((output, masks))
        loss_metric += loss.item() / (i + 1)
        dice_batch = dice(outputs, masks)
        stream_yz.set_description(
            f"Epoch:{epoch + 1}, train_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
        scheduler.step()
        accelerator.log({f"train_loss_{fold}": loss_metric, f"train_dice_batch_{fold}": dice_batch.item(),
                         f"lr_{fold}": optimizer.param_groups[0]['lr']})


def validation_fn(
        valid_loader: DataLoader,
        model: Module,
        criterion: Module,
        epoch: int,
        accelerator: Accelerator,
        fold: int,
):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    stream = tqdm(valid_loader, total=len(valid_loader), disable=not accelerator.is_local_main_process, **tqdm_style)
    loss_metric = 0
    dice_metric = []
    with torch.no_grad():
        for i, (images, masks) in enumerate(stream):
            masks = masks.float()
            images = images.float()
            output = model(images)
            loss = criterion(output, masks)
            outputs, masks = accelerator.gather((output, masks))
            loss_metric += loss.item() / (i + 1)
            outputs = outputs.sigmoid()
            outputs = outputs[:, 0, :, :] * outputs[:, 1, :, :]
            dice_batch = dice_valid(outputs, masks[:, 0, :, :])
            dice_metric.append(dice_batch.item())
            stream.set_description(
                f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
            accelerator.log({f"valid_loss_{fold}": loss_metric, f"valid_dice_batch_{fold}": dice_batch.item()})
    accelerator.print(f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice: {np.mean(dice_metric):.5f}")
    return np.mean(dice_metric)

