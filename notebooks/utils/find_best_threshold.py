from notebooks.src.metric import compute_surface_dice_score
import pandas as pd
import numpy as np
from notebooks.src.utils import rle_encode
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import yaml

solution_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_3_dense_full.csv")
from skimage import filters


def apply_hysteresis_thresholding(volume: np.array, low: float, high: float, chunk_size: int = 2):
    """
    Applies hysteresis thresholding to a 3D numpy array.

    :param volume: 3D numpy array.
    :param low: Low threshold.
    :param high: High threshold.
    :param chunk_size: Size of the chunks to process at once.
    :return: Thresholded volume.
    """
    # Apply hysteresis thresholding to each slice in the volume

    D, H, W = volume.shape
    predict = np.zeros((D, H, W), np.uint8)

    for i in range(D):
        predict[i] = np.maximum(
            filters.apply_hysteresis_threshold(volume[i], low, high),
            predict[i]
        )

    return predict


model_dir = "dm_nfnet_f2_volume_normalize_dice_find_best_epoch"
submission_df = pd.read_csv(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof_csv.csv")[
                496:997].reset_index(drop=True)
submission_df['id'] = submission_df['id'].apply(lambda x: x.replace('sparse', 'dense'))
ids = submission_df['id'].values
volume = np.load(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof_volume.npz")['volume'][496:997]
volume_threshold = apply_hysteresis_thresholding(volume, 0.2, 0.6)
rles_list = []
for i in range(len(ids)):
    rle = rle_encode(volume_threshold[i])
    rles_list.append(rle)
submission_df['rle'] = rles_list
score_final = compute_surface_dice_score(submit=submission_df, label=solution_df,
                                            )
print(score_final)
thresholds = np.linspace(0.10, 0.5, 9)
scores = {}
# for threshold in tqdm(thresholds):
#    volume_threshold = (volume > threshold).astype(np.uint8)
#    rles_list = []
#    for i in range(len(ids)):
#        rle = rle_encode(volume_threshold[i])
#        rles_list.append(rle)
#    submission_df['rle'] = rles_list
#    score_final = compute_surface_dice_score(submit=submission_df, label=solution_df,
#                                             )
#    scores[threshold] = score_final
#    print(f"Threshold: {threshold}, Score: {score_final}")
# best threshold
"""best_threshold = max(scores, key=scores.get)
print(f"Best Threshold: {best_threshold}")
print(f"Best Score: {scores[best_threshold]}")

plt.plot(list(scores.keys()), list(scores.values()))
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.show()
with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg['score'] = float(scores[best_threshold])
cfg['best_threshold'] = float(best_threshold)
with open(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/config.yaml", 'w') as f:
    yaml.dump(cfg, f, )
"""
