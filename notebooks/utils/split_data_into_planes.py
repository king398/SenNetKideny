import numpy as np
import cv2
import os
from notebooks.src.utils import rle_decode, rle_encode

import pandas as pd

volume = "kidney_3_dense"
data_dir = f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}"
kidneys_rle = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv")
# convert kidney_rle to a dict
kidneys_rle_dict = {kidneys_rle['id'][i]: kidneys_rle['kidney_rle'][i] for i in range(len(kidneys_rle))}
# only keep those rows which have kidney_1_dense in the ids
kidneys_rle = kidneys_rle[kidneys_rle['id'].str.contains(volume)].reset_index(drop=True)
masks = []
images = []
kidney_masks = []
for i in range(496, 496+len(os.listdir(f'{data_dir}/labels/'))):
    print('\r', i, end='')
    v = cv2.imread(f'{data_dir}/labels/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    masks.append(v)
    v = cv2.imread(f'{data_dir}/images/{i:04d}.tif', cv2.IMREAD_GRAYSCALE)
    images.append(v)
    kidney_masks.append(rle_decode(kidneys_rle['kidney_rle'][i], img_shape=v.shape))
masks = np.stack(masks)
images = np.stack(images)
kidney_masks = np.stack(kidney_masks)
dataset_xz = (images, masks, kidney_masks)
dataset_yz = (images, masks, kidney_masks)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/", exist_ok=True)
os.makedirs(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/", exist_ok=True)
for p in range(images.shape[1]):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/images/{p:04d}.tif",
                images[:, p])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_xz/labels/{p:04d}.tif",
                masks[:, p])
    kidneys_rle_dict.update({f"{volume}_xz_{p:04d}": rle_encode(kidney_masks[:, p])})
for p in range(images.shape[2]):
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/images/{p:04d}.tif",
                images[:, :, p])
    cv2.imwrite(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/{volume}_yz/labels/{p:04d}.tif",
                masks[:, :, p])
    kidneys_rle_dict.update({f"{volume}_yz_{p:04d}": rle_encode(kidney_masks[:, :, p])})

kidneys_rle_dict = pd.DataFrame(kidneys_rle_dict.items(), columns=['id', 'kidney_rle'])
kidneys_rle_dict.to_csv(f"/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv", index=False)
