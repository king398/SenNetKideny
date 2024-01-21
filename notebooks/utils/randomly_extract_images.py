import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import glob
from tqdm import tqdm
import torch
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets


files = sorted(glob.glob("/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_1_dense/images/*.tif"))
volume = np.stack([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in tqdm(files)])
masks = np.stack([cv2.imread(i.replace("images", "labels"), cv2.IMREAD_GRAYSCALE) for i in tqdm(files)])

images, masks = cutmix(torch.from_numpy(volume[699:704]).un-squeeze(1).float(), torch.from_numpy(masks[699:704]).unsqueeze(1).float(), 1)
images = images.squeeze(1).numpy()
masks = masks[0].squeeze(1).numpy()
plt.imshow(images[3])
plt.show()