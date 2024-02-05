import re
import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group()) if match else None


def apply_hysteresis_thresholding(volume: np.array, low: float, high: float, chunk_size: int = 32):
    D, H, W = volume.shape
    predict = np.zeros((D, H, W), np.uint8)

    for i in range(0, D, chunk_size // 2):
        predict[i:i + chunk_size] = np.maximum(
            apply_hysteresis_threshold(volume[i:i + chunk_size], low, high),
            predict[i:i + chunk_size]
        )
    return predict


def choose_biggest_object(mask, threshold):
    mask = ((mask > threshold) * 255).astype(np.uint8)
    num_label, mask, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_label, max_area = -1, -1
    for label in range(1, num_label):
        if stats[label, cv2.CC_STAT_AREA] >= max_area:
            max_area = stats[label, cv2.CC_STAT_AREA]
            max_label = label
    return (mask == max_label).astype(np.uint8)


def rle_encode(mask: np.array) -> str:
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle