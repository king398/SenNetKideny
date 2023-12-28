from tqdm.auto import tqdm
import gc
import gc
from metric import compute_surface_dice_score
import pandas as pd
from accelerate import Accelerator
from torch import optim
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import CombinedDataLoader
from utils import *

dice = Dice()
dice_valid = Dice_Valid()


def get_color_escape(r, g, b, background=False):
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'


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
    tqdm_color = get_color_escape(255, 0, 0)  # Red color for example
    tqdm_style = {
        'bar_format': f'{tqdm_color}{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]{get_color_escape(255, 255, 255)}'}
    train_loader = CombinedDataLoader(train_loader, train_loader_xz, train_loader_yz)
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
    pd_dataframes = [{"id": [], "rle": []} for _ in range(5)]
    labels_df = {"id": [], "rle": []}
    j = 0
    x = 0

    with torch.no_grad():
        for i, (images, masks) in enumerate(stream):
            masks = masks.float()
            images = images.float()
            output = model(images, inference=True)
            loss = criterion(output, masks)
            outputs, masks = accelerator.gather((output, masks))
            loss_metric += loss.item() / (i + 1)
            outputs = outputs.sigmoid()
            outputs_not_multiply = outputs.detach().clone()
            outputs = outputs[:, 0, :, :] * outputs[:, 1, :, :]
            dice_batch = dice_valid(outputs, masks[:, 0, :, :])
            dice_metric.append(dice_batch.item())
            stream.set_description(
                f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")
            outputs_not_multiply = outputs_not_multiply.detach().cpu().float().numpy()
            masks = masks.detach().cpu().float().numpy()
            for p, image in enumerate(outputs_not_multiply):
                kidney = image[1, :, :]
                kidney = choose_biggest_object(kidney, 0.5)
                output_mask = image[0, :, :] * kidney
                # iterate from threshold 0.1 to 0.5
                threshold = [0.1, 0.2, 0.3, 0.4, 0.5]
                for m, t in enumerate(threshold):
                    output_mask_new = (output_mask > t).astype(np.uint8)
                    rle_mask = rle_encode(output_mask_new)
                    pd_dataframes[m]["id"].append(f"kidney_3_dense_{j:04d}")
                    pd_dataframes[m]["rle"].append(rle_mask)
                j += 1
            for mask in masks[:, 0, :, :]:
                rle_mask = rle_encode(mask)
                labels_df["id"].append(f"kidney_3_dense_{x:04d}")
                labels_df["rle"].append(rle_mask)
                x += 1

            accelerator.log({f"valid_loss_{fold}": loss_metric, f"valid_dice_batch_{fold}": dice_batch.item()})
    labels_df = pd.DataFrame(labels_df)
    labels_df['width'] = 1510
    labels_df['height'] = 1706
    labels_df['group'] = 'kidney_3_dense'
    labels_df['slice'] = np.arange(len(labels_df))
    threshold_score_dict = {}
    for m, pd_dataframe in enumerate(pd_dataframes):
        m += 1
        pd_dataframe = pd.DataFrame(pd_dataframe)

        surface_dice = compute_surface_dice_score(submit=pd_dataframe, label=labels_df)
        threshold_score_dict.update({f"threshold_{m / 10}": surface_dice})
    max_surface_dice = max(threshold_score_dict.values())
    best_threshold = list(threshold_score_dict.keys())[list(threshold_score_dict.values()).index(max_surface_dice)]
    accelerator.print(
        f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}, dice: {np.mean(dice_metric):.5f} ,surface_dice: {max_surface_dice:.5f} ,threshold_score_dict   {threshold_score_dict} ")
    accelerator.log({f"surface_dice": max_surface_dice})

    return np.mean(dice_metric)
