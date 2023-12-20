from notebooks.src.metric import compute_surface_dice_score
import pandas as pd
import numpy as np

solution_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_3_dense_full.csv")

solution_df['width'] = 1510
solution_df['height'] = 1706
solution_df['group'] = 'kidney_3_dense'
solution_df['slice'] = np.arange(len(solution_df))

model_dir = "seresnext101d_32x8d_pad_kidney_multiview"
submission_df = pd.read_csv(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof.csv")
# replace sparse with dense
submission_df['id'] = submission_df['id'].apply(lambda x: x.replace('sparse', 'dense'))
# only keep the id with are present in solution
submission_df = submission_df[submission_df['id'].isin(solution_df['id'].values)].reset_index(drop=True)
score_final = compute_surface_dice_score(submit=submission_df, label=solution_df,
                                         )
import yaml

with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg['score'] = float(score_final)
with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml", 'w') as f:
    yaml.dump(cfg, f, )

print(score_final)
