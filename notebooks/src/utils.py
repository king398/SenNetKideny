import os
import cv2
import yaml
import torch
import random
import accelerate
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm
from skimage import filters
from joblib import Parallel, delayed
from segmentation_models_pytorch.utils.metrics import Fscore


def get_color_escape(r, g, b, background=False):
    return f'\033[{"48" if background else "38"};2;{r};{g};{b}m'


class Dice(nn.Module):
    def __init__(self, threshold=0.5):
        super(Dice, self).__init__()
        self.metric = Fscore(threshold=threshold, activation='sigmoid')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        dice = self.metric(inputs, targets)

        return dice


class Dice_Valid(nn.Module):
    def __init__(self, threshold=0.5):
        super(Dice_Valid, self).__init__()
        self.metric = Fscore(threshold=threshold, )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        dice = self.metric(inputs, targets)

        return dice


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    accelerate.utils.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


def write_yaml(config: dict, save_path: str) -> None:
    with open(save_path, 'w') as f:
        yaml.dump(config, f, )


def rle_encode(mask: np.array):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle


def rle_decode(mask_rle: str, shape: tuple) -> np.array:
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def apply_canny_threshold_in_chunk(x: np.array, low: float, high: float, chunk_size: int = 32):
    D, H, W = x.shape
    predict = np.zeros((D, H, W), np.uint8)
    # Modify the iteration to step fully by chunk_size without overlapping
    for i in range(0, D, chunk_size):
        # Determine the end index of the chunk, considering the end of the array
        end_i = i + chunk_size if i + chunk_size < D else D
        # Extract the chunk of images to process with Canny
        chunk = x[i:end_i]
        # Apply Canny edge detection on each 2D slice in the chunk
        chunk_edges = np.zeros_like(chunk)
        for j in range(chunk.shape[0]):
            chunk_edges[j] = cv2.Canny(chunk[j].astype(np.uint8), low, high)
        # Place the processed chunk back into the predict array
        predict[i:end_i] = chunk_edges
    return predict


def remove_small_objects(mask, min_size):
    # Find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= min_size:
            processed[label == l] = 255
    return processed


def choose_biggest_object(mask, threshold):
    mask = ((mask > threshold) * 255).astype(np.uint8)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_label = -1
    max_area = -1
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= max_area:
            max_area = stats[l, cv2.CC_STAT_AREA]
            max_label = l
    processed = (label == max_label).astype(np.uint8)
    return processed


def apply_hysteresis_thresholding(volume: np.array, low: float, high: float, chunk_size: int = 32):
    D, H, W = volume.shape
    predict = np.zeros((D, H, W), np.uint8)

    for i in range(0, D, chunk_size // 2):
        predict[i:i + chunk_size] = np.maximum(
            filters.apply_hysteresis_threshold(volume[i:i + chunk_size], low, high),
            predict[i:i + chunk_size]
        )
    return predict


min_max = {
    "kidney_1_dense": {
        "mean": 23165.613198187508,
        "std": 2747.315335558582,
        "min": 0.0,
        "max": 65535.0,
        "q_min": 20243.0,
        "q_max": 29649.0
    },
    "kidney_2": {
        "mean": 32800.428034835626,
        "std": 2534.013389861678,
        "min": 2928.0,
        "max": 65535.0,
        "q_min": 29784.0,
        "q_max": 42380.0
    },
    "kidney_3_dense": {
        "mean": 19570.52992715221,
        "std": 758.3516133619929,
        "min": 4903.0,
        "max": 63208.0,
        "q_min": 18806.0,
        "q_max": 21903.0
    },
    "kidney_3_sparse": {
        "mean": 19536.406272288794,
        "std": 881.0674124854231,
        "min": 1488.0,
        "max": 65248.0,
        "q_min": 18966.0,
        "q_max": 21944.0
    }
}


def norm_by_percentile(volume, low=10, high=99.8, alpha=0.01):
    q_min = np.percentile(volume, low)
    q_max = np.percentile(volume, high)
    x = (volume - q_min) / (q_max - q_min)
    x[x > 1] = (x[x > 1] - 1) * alpha + 1
    x[x < 0] = (x[x < 0]) * alpha
    return x


def read_cv2(fp):
    return cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float16)


def load_images_and_masks(cfg, kidney_name: str):
    csv = pd.read_csv(cfg['csv_image_mask_paths'])
    if "xz" in kidney_name or "yz" in kidney_name:
        csv = csv[csv.id.str.contains(kidney_name)]
        images_paths = csv.img_path.tolist()
        masks_paths = csv.mask_path.tolist()
        kid_masks_paths = csv.rle_path.tolist()
        return images_paths, masks_paths, kid_masks_paths
    else:
        csv = csv[csv.id.str.contains(kidney_name) & ~csv.id.str.contains('xz|yz')]
        images_paths = csv.img_path.tolist()
        masks_paths = csv.mask_path.tolist()
        kid_masks_paths = csv.rle_path.tolist()

        desc = 'reading and stacking images in volume...'
        if kidney_name in cfg['cub1_path_percentile']:
            # volume = np.load(cfg['cub1_path_percentile']).squeeze(3).transpose(2, 0, 1)
            volume = np.load(cfg['cub1_path_percentile']).transpose(2, 0, 1)
            volume = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        else:
            # volume = np.load(cfg['cub3_path_percentile']).squeeze(3).transpose(2, 0, 1)
            volume = np.load(cfg['cub3_path_percentile']).transpose(2, 0, 1)
            volume = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # volume = np.stack([read_cv2(fp) for fp in tqdm(images_paths, total=len(images_paths), desc=desc)])
        return images_paths, masks_paths, kid_masks_paths, volume
