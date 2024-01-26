import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from tqdm import tqdm
import torch
import random
from torchvision.transforms import v2


def make_digonal_slice(shape=[100, 120, 150]):  # D, H, W
    D, H, W = shape

    S = max(H, W)
    L = min(H, W)
    L = int(0.25 * ((L ** 2 + L ** 2) ** 0.5))

    x, y = np.arange(S), np.arange(S)

    # diag = np.stack([np.arange(S),np.arange(S)]) #shape (2, 150)
    # diagy = diag + np.stack([np.arange(S),np.zeros(S)])

    coord1 = []
    coord2 = []
    for h in range(H):
        xx = x
        yy = y + h
        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if valid.sum() > L:
            yy1, xx1 = yy[valid], xx[valid]
            coord1.append([yy1, xx1])

            yy2 = yy1
            xx2 = W - 1 - xx1  # np.ascontiguousarray(xx1[::-1])
            coord2.append([yy2, xx2])

    for w in range(1, W):
        xx = x + w
        yy = y
        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if valid.sum() > L:
            yy1, xx1 = yy[valid], xx[valid]
            coord1.append([yy1, xx1])

            yy2 = yy1
            xx2 = W - 1 - xx1
            coord2.append([yy2, xx2])

    coord = coord1 + coord2
    return coord


files = glob.glob("/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_1_dense/images/*.tif")
random.shuffle(files)
volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in tqdm(files)])

coord = make_digonal_slice(shape=volume.shape)
counter = 0
# example usage
for y, x in coord:
    slice = volume[:, y, x]
    counter += 1

