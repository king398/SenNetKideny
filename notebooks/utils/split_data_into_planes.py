import numpy as np
import cv2
import os
from notebooks.src.utils import rle_decode, rle_encode

import pandas as pd

volume = "kidney_1_dense"
data_dir = f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}"
kidneys_rle = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv")
# convert kidney_rle to a dict
kidneys_rle_dict = {kidneys_rle['id'][i]: kidneys_rle['kidney_rle'][i] for i in range(len(kidneys_rle))}
# only keep those rows which have kidney_1_dense in the ids
kidneys_rle = kidneys_rle[kidneys_rle['id'].str.contains('kidney_1_dense')].reset_index(drop=True)
masks = []
images = []
kidney_masks = []
for i in range(len(os.listdir(f'{data_dir}/labels/'))):
    print('\r', i, end='')
    v = cv2.imread(f'{data_dir}/labels/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    masks.append(v)
    v = cv2.imread(f'{data_dir}/images/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    images.append(v)
    kidney_masks.append(rle_decode(kidneys_rle['kidney_rle'][i], img_shape=v.shape))
masks = np.stack(masks)
images = np.stack(images)
kidney_masks = np.stack(kidney_masks)
dataset_xz = (images.transpose(1, 2, 0), masks.transpose(1, 2, 0), kidney_masks.transpose(1, 2, 0))
dataset_yz = (images.transpose(2, 0, 1), masks.transpose(2, 0, 1), kidney_masks.transpose(2, 0, 1))
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/", exist_ok=True)
for i in range(len(dataset_xz[0])):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/{i:04d}.tif",
                dataset_xz[0][i])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/{i:04d}.tif",
                dataset_xz[1][i])
    kidneys_rle_dict.update({f"{volume}_xz_{i:04d}": rle_encode(dataset_xz[2][i])})
for i in range(len(dataset_yz[0])):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/{i:04d}.tif",
                dataset_yz[0][i])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/{i:04d}.tif",
                dataset_yz[1][i])
    kidneys_rle_dict.update({f"{volume}_yz_{i:04d}": rle_encode(dataset_yz[2][i])})

kidneys_rle_dict = pd.DataFrame(kidneys_rle_dict.items(), columns=['id', 'kidney_rle'])
kidneys_rle_dict.to_csv(f"/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv", index=False)
