from notebooks.src.metric import fast_compute_surface_dice_score_from_tensor
from notebooks.src.utils import rle_decode, rle_encode
import pandas as pd
import numpy as np

solution_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/solution_df.csv")
# only keep those which have kidney_3_sparse in them
solution_df = solution_df[solution_df['id'].str.contains("kidney_3_sparse")]
solution_df['width'] = 1510
solution_df['height'] = 1706
solution_df['group'] = 'kidney_3_dense'
solution_df['slice'] = np.arange(len(solution_df))

model_dir = "seresnext101d_32x8d_pad_kidney_multiview"
volume  = np.load(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/volume.npz")['volume']
# volume = np.transpose(volume, (0, 2, 1))
truth = np.stack([rle_decode(solution_df["rle"].values[i], (1706, 1510)) for i in range(len(solution_df))])
# replace sparse with dense
# submission_df['id'] = submission_df['id'].apply(lambda x: x.replace('sparse', 'dense'))
# only keep the id with are present in solution
# submission_df = submission_df[submission_df['id'].isin(solution_df['id'].values)]
score_final = fast_compute_surface_dice_score_from_tensor(predict=volume, truth=truth)
import yaml

with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg['score'] = float(score_final)
with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml", 'w') as f:
    yaml.dump(cfg, f, )

print(score_final)
