import torch
from tqdm.auto import tqdm
import gc
from torch.utils.data import DataLoader
from utils import *
from torch.nn import Module
from torch import optim
from accelerate import Accelerator
import pandas as pd
from metric import compute_surface_dice_score
from dataset import CombinedDataLoader

dice = Dice()
dice_valid = Dice_Valid()

tqdm_color = get_color_escape(0, 229, 255)  # Red color for example
tqdm_style = {
    'bar_format': f'{tqdm_color}{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}},'
                  f' {{rate_fmt}}{{postfix}}]{get_color_escape(255, 255, 255)}'}


def train_fn(
        data_loader_list: list,
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
    combined_loader = CombinedDataLoader(*data_loader_list)
    stream = tqdm(combined_loader, total=len(combined_loader), disable=not accelerator.is_local_main_process,
                  **tqdm_style)

    for i, (images, masks, image_ids) in enumerate(stream):
        with accelerator.accumulate(model):
            masks = masks.float().contiguous()
            images = images.float().contiguous()
            output = model(images)
            loss = criterion(output, masks)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            outputs, masks = accelerator.gather_for_metrics((output, masks))
            loss_metric += loss.item() / (i + 1)
            dice_batch = dice(outputs, masks)
            stream.set_description(
                f"Epoch:{epoch + 1}, train_loss: {loss_metric:.5f}, dice_batch: {dice_batch.item():.5f}")

            accelerator.log({f"train_loss_{fold}": loss_metric, f"train_dice_batch_{fold}": dice_batch.item(),
                             f"lr_{fold}": optimizer.param_groups[0]['lr']})


def validation_fn(
        valid_loader: DataLoader,
        model: Module,
        criterion: Module,
        epoch: int,
        accelerator: Accelerator,
        labels_df: pd.DataFrame,
        model_dir: str,
):
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    stream = tqdm(valid_loader, total=len(valid_loader), disable=not accelerator.is_local_main_process, )
    loss_metric = 0
    pd_dataframes = [{"id": [], "rle": []} for _ in range(5)]
    j = 0
    x = 0
    dice_list = []
    with torch.no_grad():
        for i, (images, masks, image_ids) in enumerate(stream):

            masks = masks.float()
            images = images.float().to(accelerator.device)

            output = model(images, )
            loss = criterion(output, masks)
            outputs, masks, = accelerator.gather((output, masks,))
            image_ids = accelerator.gather_for_metrics(image_ids)
            loss_metric += loss.item() / (i + 1)
            outputs = outputs.sigmoid()
            outputs_not_multiply = outputs.detach().clone()
            stream.set_description(
                f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f}")
            outputs_not_multiply = outputs_not_multiply.detach().cpu().float().numpy()
            outputs = outputs[:, 0, :, :] * outputs[:, 1, :, :]
            dice_batch = dice_valid(outputs, masks[:, 0, :, :])
            dice_list.append(dice_batch.item())

            for p, image, in enumerate(outputs_not_multiply, ):
                kidney = image[1, :, :]
                kidney = choose_biggest_object(kidney, 0.5)
                output_mask = image[0, :, :] * kidney
                # iterate from threshold 0.1 to 0.5
                threshold = [0.1, 0.2, 0.3, 0.4, 0.5]

                for m, t in enumerate(threshold):
                    output_mask_new = output_mask.copy()
                    output_mask_new = (output_mask_new > t).astype(np.uint8)
                    rle_mask = rle_encode(output_mask_new)
                    pd_dataframes[m]["id"].append(image_ids[p])
                    pd_dataframes[m]["rle"].append(rle_mask)
                j += 1

    threshold_score_dict = {}
    for m, pd_dataframe in enumerate(pd_dataframes):
        m += 1
        pd_dataframe = pd.DataFrame(pd_dataframe)
        #     # drop all duplicates in the dataframe
        pd_dataframe = pd_dataframe.drop_duplicates(subset=['id'])
        surface_dice = compute_surface_dice_score(submit=pd_dataframe, label=labels_df)
        threshold_score_dict.update({f"threshold_{m / 10}": surface_dice})
    max_surface_dice = max(threshold_score_dict.values())
    best_threshold = list(threshold_score_dict.keys())[list(threshold_score_dict.values()).index(max_surface_dice)]
    dice_score = np.mean(dice_list)

    accelerator.print(
        f"Epoch:{epoch + 1}, valid_loss: {loss_metric:.5f} ,Dice Coefficient {dice_score},surface_dice: {max_surface_dice:.5f} ,threshold_score_dict   {threshold_score_dict} ")
    accelerator.log(
        {f"surface_dice": max_surface_dice, f"valid_loss": loss_metric, f"best_threshold": best_threshold, })
    return dice_score, max_surface_dice


