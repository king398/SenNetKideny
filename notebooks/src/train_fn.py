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
dice = Dice()


def train_fn(
        train_loader: DataLoader,
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
    stream = tqdm(train_loader, total=len(train_loader), disable=not accelerator.is_local_main_process, )

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
    stream = tqdm(valid_loader, total=len(valid_loader), disable=not accelerator.is_local_main_process, )
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
            dice_batch = dice(outputs, masks)
            dice_metric.append(dice_batch.item())
            stream.set_description(
                f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
            accelerator.log({f"valid_loss_{fold}": loss_metric, f"valid_dice_batch_{fold}": dice_batch.item()})
    accelerator.print(f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice: {np.mean(dice_metric):.5f}")
    return np.mean(dice_metric)
def oof_fn(model: nn.Module, data_loader: DataLoader, device: torch.device, ):
    torch.cuda.empty_cache()
    model.eval()
    rles_list = []
    image_ids_all = []
    for i, (images, image_shapes, image_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device, non_blocking=True).float()
        with torch.no_grad() and autocast():
            outputs = model(images).sigmoid().detach().cpu().float()
        for i, image in enumerate(outputs):
            output_mask = F.interpolate(image.unsqueeze(0),
                                        size=(int(image_shapes[0][i]), int(image_shapes[1][i]))).squeeze()
            output_mask = (output_mask > 0.25).float().numpy()
            output_mask *= 255
            output_mask = output_mask.astype(np.uint8)
            output_mask = remove_small_objects(output_mask, 10)
            rle_mask = rle_encode(output_mask)
            rles_list.append(rle_mask)
            image_ids_all.append(image_ids[i])
        del outputs, images, output_mask
        gc.collect()

    return rles_list, image_ids_all
