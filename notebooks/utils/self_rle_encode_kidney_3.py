import cv2
import pandas as pd
from notebooks.src.utils import rle_encode
import os
import numpy as np

label_df = pd.DataFrame(columns=['id', 'rle'])
labels = sorted(os.listdir("/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_3_dense/labels"))
for label in labels:
    image = cv2.imread(f"/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_3_dense/labels/{label}", )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    label_df = label_df.append({'id': f'kidney_3_dense_{label.replace(".tif", "")}', 'rle': rle_encode(image)},
                               ignore_index=True)
label_df['width'] = 1510
label_df['height'] = 1706
label_df['group'] = 'kidney_3_dense'
label_df['slice'] = np.arange(len(label_df))
label_df.to_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_3_dense_full_self_encode.csv", index=False)
