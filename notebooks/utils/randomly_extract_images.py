import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from tqdm import tqdm
import torch
import random
from torchvision.transforms import v2


def CutMix(images: torch.tensor, masks: torch.tensor):
    y = torch.randint(4, (images.shape[0],), dtype=torch.long)
    concat = torch.cat([images, masks], dim=1)
    cutmix = v2.CutMix(num_classes=4)
    cutmixed_images_masks, _ = cutmix(concat, y)
    return cutmixed_images_masks[:, 0:3], cutmixed_images_masks[:, 3:]


files = glob.glob("/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_1_dense/images/*.tif")
random.shuffle(files)
volume = np.stack([cv2.imread(i) for i in tqdm(files)])
masks_volume = np.stack([cv2.imread(i.replace("images", "labels"), cv2.IMREAD_GRAYSCALE) for i in tqdm(files)])

images, masks = CutMix(torch.from_numpy(volume[699:704]).float().permute(0, 3, 1, 2),
                       torch.from_numpy(masks_volume[699:704]).unsqueeze(1).float())
images = images.permute(0, 2, 3, 1).numpy()
masks = masks[0].permute(1,2,0).numpy()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(images[0] / 255)
ax[1].imshow(masks)
plt.show()

