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
import pandas as pd
from metric import compute_surface_dice_score

dice = Dice()
dice_valid = Dice_Valid()

tqdm_color = get_color_escape(0, 229, 255)  # Red color for example
tqdm_style = {
    'bar_format': f'{tqdm_color}{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}},'
                  f' {{rate_fmt}}{{postfix}}]{get_color_escape(255, 255, 255)}'}


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
    for i, (images, masks, original_shape) in enumerate(stream):
        break
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

    for i, (images, masks, original_shape) in enumerate(stream_xz):
        break
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
    for i, (images, masks, original_shape) in enumerate(stream_yz):
        break
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
        validation_df: pd.DataFrame
):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    stream = tqdm(valid_loader, total=len(valid_loader), disable=not accelerator.is_local_main_process, **tqdm_style)
    loss_metric = 0
    pd_dataframe = {"id": [], "rle": []}
    j = 0
    with torch.no_grad():
        for i, (images, masks, original_shape) in enumerate(stream):
            masks = masks.float()
            images = images.float()
            output = model(images)
            loss = criterion(output, masks)
            outputs, masks = accelerator.gather((output, masks))
            loss_metric += loss.item() / (i + 1)
            outputs = accelerator.gather_for_metrics(outputs)
            outputs = outputs.sigmoid().detach().cpu().float().numpy()
            # outputs = outputs[:, 0, :, :] * outputs[:, 1, :, :]
            # dice_batch = dice_valid(outputs, masks[:, 0, :, :])
            stream.set_description(
                f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, ")
            for image in outputs:
                if f"kidney_3_dense_{j:04d}" not in validation_df['id'].values:
                    j += 1
                    continue
                kidney = image[1, :, :]
                kidney = choose_biggest_object(kidney, 0.5)
                output_mask = image[0, :, :] * kidney
                output_mask = (output_mask > 0.15).astype(np.uint8)
                rle_mask = rle_encode(output_mask)
                pd_dataframe["id"].append(f"kidney_3_dense_{j:04d}")
                pd_dataframe["rle"].append(rle_mask)
                j += 1

            accelerator.log({f"valid_loss_{fold}": loss_metric})
    # drop all the rows from pd_dataframe which ids are not present in validation_df
    pd_dataframe = pd.DataFrame(pd_dataframe)
    surface_dice = compute_surface_dice_score(submit=pd_dataframe, label=validation_df)
    accelerator.print(f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f} surface_dice: {surface_dice:.5f}")
    accelerator.log({f"surface_dice_{fold}": surface_dice})
    return surface_dice


def oof_fn(model: nn.Module, data_loader: DataLoader, data_loader_xz: DataLoader, data_loader_yz: DataLoader,
           device: torch.device, volume_shape, ):
    torch.cuda.empty_cache()
    model.eval()
    volume = np.zeros(volume_shape)
    print(volume.shape)
    print(volume[:, 0, :].shape)
    print(volume[:, :, 0].shape)
    rles_list = []
    image_ids_all = []
    j = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):

        # tranpose the image
        images = images.to(device, non_blocking=True).float()
        with torch.no_grad() and autocast():
            outputs = model(images).sigmoid().detach().cpu().float()

            # outputs = outputs[:, 0, :, :] * outputs[:, 1, :, :]
        for p, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :]
            output_mask = (output_mask.numpy() * kidney)

            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][p]),
                                          original_width=int(image_shapes[1][p]))
            volume[j] += output_mask

            image_ids_all.append(image_ids[p])
            j += 1
        del outputs, images, output_mask
    j = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_xz), total=len(data_loader_xz)):

        images = images.to(device, non_blocking=True).float()
        with torch.no_grad() and autocast():
            outputs = model(images).sigmoid().detach().cpu().float()

        for p, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :]
            output_mask = (output_mask.numpy() * kidney)
            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][p]),
                                          original_width=int(image_shapes[1
                                                             ][p]))

            volume[:, j] += output_mask
            j += 1
    j = 0
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader_yz), total=len(data_loader_yz)):

        if i == 0:
            print(images.shape)
        images = images.to(device, non_blocking=True).float()
        with torch.no_grad() and autocast():
            outputs = model(images).sigmoid().detach().cpu().float()
        for p, image in enumerate(outputs):
            kidney = image[1, :, :]
            kidney = choose_biggest_object(kidney.numpy(), 0.5)
            output_mask = image[0, :, :] * kidney
            output_mask = (output_mask.numpy() * kidney)
            output_mask = reverse_padding(output_mask,
                                          original_height=int(image_shapes[0][p]),
                                          original_width=int(image_shapes[1][p]))

            volume[:, :, j] += output_mask
            j += 1
        gc.collect()

    volume = volume / 3
    volume = ((volume > 0.2) * 255).astype(np.uint8)
    print(volume.max())
    for output_mask in volume:
        rle_mask = rle_encode(output_mask)
        rles_list.append(rle_mask)
    print(volume.shape)
    return rles_list, image_ids_all, volume
