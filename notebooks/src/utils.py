from segmentation_models_pytorch.utils.metrics import Fscore
from torch import nn
import torch
import os
import numpy as np
import random
import accelerate
import yaml
import cv2
from skimage import filters


class Dice(nn.Module):
    def __init__(self, threshold=0.5):
        super(Dice, self).__init__()
        self.metric = Fscore(threshold=threshold, activation='sigmoid')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        dice = self.metric(inputs, targets)

        return dice


def norm_by_percentile(volume, low=10, high=99.8, alpha=0.01):
    xmin = np.percentile(volume, low)
    xmax = np.percentile(volume, high)
    x = (volume - xmin) / (xmax - xmin)
    if 1:
        x[x > 1] = (x[x > 1] - 1) * alpha + 1
        x[x < 0] = (x[x < 0]) * alpha
    # x = np.clip(x,0,1)
    return x


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


def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 1
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    return img


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
    """
    Applies hysteresis thresholding to a 3D numpy array.

    :param volume: 3D numpy array.
    :param low: Low threshold.
    :param high: High threshold.
    :param chunk_size: Size of the chunks to process at once.
    :return: Thresholded volume.
    """
    # Apply hysteresis thresholding to each slice in the volume

    D, H, W = volume.shape
    predict = np.zeros((D, H, W), np.uint8)

    for i in range(0, D, chunk_size // 2):
        predict[i:i + chunk_size] = np.maximum(
            filters.apply_hysteresis_threshold(volume[i:i + chunk_size], low, high),
            predict[i:i + chunk_size]
        )

    return predict
