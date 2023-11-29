import gc

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple, Any
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
import random
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import torch.nn.functional as F
import glob
from torch import nn
import pandas as pd


# https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/456033
def remove_small_objects(mask, min_size):
    # Find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= min_size:
            processed[label == l] = 255

    return processed


def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle


def get_valid_transform(DIM) -> Compose:
    return Compose([
        Resize(DIM, DIM),
        ToTensorV2(),
    ])


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


class ImageDataset(Dataset):
    def __init__(self, image_paths: list, transform: Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, item) -> tuple[torch.Tensor, tuple[str, ...], str]:
        image = cv2.imread(self.image_paths[item])
        image_id = self.image_paths[item].split("/")[-1].split(".")[0]
        folder_id = self.image_paths[item].split("/")[-3]
        image_id = f"{folder_id}_{image_id}"
        image_shape = image.shape
        image_shape = tuple(str(element) for element in image_shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - image.min()) / (image.max() - image.min() + 0.0001)
        augmented = self.transform(image=image)
        image = augmented["image"]
        return image, image_shape, image_id


def return_model(model_name: str, in_channels: int, classes: int):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,

    )
    return model


def inference_fn(model: nn.Module, data_loader: DataLoader, device: torch.device, ):
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
            output_mask = (output_mask > 0.5).float().numpy()
            output_mask *= 255
            output_mask = output_mask.astype(np.uint8)
            output_mask = remove_small_objects(output_mask, 50)
            rle_mask = rle_encode(output_mask)
            rles_list.append(rle_mask)
            image_ids_all.append(image_ids[i])
        del outputs, images, output_mask
        gc.collect()

    return rles_list, image_ids_all


def main(cfg: dict):
    seed_everything(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_files = sorted(glob.glob(f"{cfg['test_dir']}/*/images/*.tif"))
    model = return_model(cfg['model_name'], cfg['in_channels'], cfg['classes'])
    model.load_state_dict(torch.load(cfg["model_path"], map_location=torch.device('cpu')), )
    model.to(device)
    test_dataset = ImageDataset(test_files, get_valid_transform(cfg['image_size']))
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)
    rles_list, image_ids = inference_fn(model, test_loader, device)
    submission = pd.DataFrame()
    submission['id'] = image_ids
    submission['rle'] = rles_list
    submission.to_csv('submission.csv', index=False)
    print(submission.head())


config = {
    "seed": 42,
    "model_name": "resnet50",
    "in_channels": 3,
    "classes": 1,
    "test_dir": '/kaggle/input/blood-vessel-segmentation/test',
    "model_path": "/kaggle/input/senet-models/resnet50_baseline/model.pth",
    "image_size": 1536,
    "batch_size": 2,
    "num_workers": 2,

}
if __name__ == "__main__":
    main(config)
