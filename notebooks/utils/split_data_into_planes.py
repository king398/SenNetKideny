import numpy as np
import cv2
import os

volume = "kidney_1_dense"
data_dir = f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}"

masks = []
images = []
for i in range(len(os.listdir(f'{data_dir}/labels/'))):
    print('\r', i, end='')
    v = cv2.imread(f'{data_dir}/labels/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    masks.append(v)
    v = cv2.imread(f'{data_dir}/images/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    images.append(v)
masks = np.stack(masks)
images = np.stack(images)
dataset_xz = (images.transpose(1, 2, 0), masks.transpose(1, 2, 0))
dataset_yz = (images.transpose(2, 0, 1), masks.transpose(2, 0, 1))
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/", exist_ok=True)
for i in range(len(dataset_xz[0])):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/{i:04d}.tif", dataset_xz[0][i])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/{i:04d}.tif", dataset_xz[1][i])
for i in range(len(dataset_yz[0])):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/{i:04d}.tif", dataset_yz[0][i])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/{i:04d}.tif", dataset_yz[1][i])

