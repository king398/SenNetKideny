import numpy as np
from notebooks.src.utils import *
import pandas as pd
from matplotlib import pyplot as plt
import cv2

df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/models/seresnext26d_32x4d_pad_kidney_2/oof.csv")
train_rles_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles.csv")
train_rles_dict = {train_rles_df['id'][i]: train_rles_df['rle'][i] for i in range(len(train_rles_df))}
index = 600
rle = df['rle'][index]
id = df['id'][index]
mask_truth = rle_decode(train_rles_dict[id], (1706, 1510)) * 255

mask = rle_decode(rle, (1706, 1510)) * 255
print(mask.max())
cv2.imshow( "mask",mask)
cv2.imshow( "mask_truth",mask_truth)
cv2.waitKey(0)
