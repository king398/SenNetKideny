import pandas as pd
import numpy as np

df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv")
# only keep those rows which have kidney_2 in them
df = df[df['id'].str.contains("kidney_2")].reset_index(drop=True)
df['width'] = 1511
df['height'] = 1041
df['group'] = 'kidney_2'
df['slice'] = np.arange(len(df))
print(df.head())
df.to_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_2_dense.csv", index=False)
