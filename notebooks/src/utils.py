import gc

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
from tqdm import tqdm


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
    xmin = np.percentile(volume, low)
    xmax = np.percentile(volume, high)
    x = (volume - xmin) / (xmax - xmin)
    x[x > 1] = (x[x > 1] - 1) * alpha + 1
    x[x < 0] = (x[x < 0]) * alpha
    # x = np.clip(x,0,1)
    return x


def load_images_and_masks(directory: str, image_subdir: str, label_subdir: str, kidney_rle: dict,
                          kidney_rle_prefix: str):
    image_dir = os.path.join(directory, image_subdir)
    label_dir = os.path.join(directory, label_subdir)
    image_files = sorted(os.listdir(image_dir))
    if kidney_rle_prefix == 'kidney_2':
        image_files = image_files[900:]

    images_full_path = [os.path.join(image_dir, f) for f in image_files]
    labels_full_path = [f.replace(image_subdir, label_subdir) for f in images_full_path]

    kidneys_rle = [kidney_rle[f"{kidney_rle_prefix}_{f.split('.')[0]}"] for f in image_files]
    if "xz" in directory or "yz" in directory:
        return images_full_path, labels_full_path, kidneys_rle
    else:
        volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE).astype(np.float16) for i in tqdm(images_full_path)])

        return images_full_path, labels_full_path, kidneys_rle, norm_by_percentile(volume)
def load_images_and_masks_pseudo(directory: str, image_subdir: str):
    image_dir = os.path.join(directory, image_subdir)
    image_files = sorted(os.listdir(image_dir))

    images_full_path = [os.path.join(image_dir, f) for f in image_files]
    masks = np.load(f"{directory}/labels/volume.npy").astype(np.float16)
    volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE).astype(np.float16) for i in tqdm(images_full_path)])

    return images_full_path, norm_by_percentile(volume), masks