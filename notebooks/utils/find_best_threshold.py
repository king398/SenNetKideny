from notebooks.src.metric import compute_surface_dice_score
import pandas as pd
import numpy as np
from notebooks.src.utils import rle_encode
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

solution_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/kidney_2_solution.csv")

model_dir = "maxvit_small_tf_multiview_15_epoch_5e_04_retry_validation"
submission_df = pd.read_csv(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof_csv.csv")
ids = submission_df['id'].values
volume = np.load(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof_volume.npz")['volume']
thresholds = np.linspace(0.10, 0.9, 9)
scores = {}
for threshold in tqdm(thresholds):
    volume_threshold = (volume > threshold).astype(np.uint8)
    rles_list = []
    for i in range(len(ids)):
        rle = rle_encode(volume_threshold[i])
        rles_list.append(rle)
    submission_df['rle'] = rles_list
    score_final = compute_surface_dice_score(submit=submission_df, label=solution_df,
                                             )
    scores[threshold] = score_final
    print(f"Threshold: {threshold}, Score: {score_final}")
# best threshold
best_threshold = max(scores, key=scores.get)
print(f"Best Threshold: {best_threshold}")
print(f"Best Score: {scores[best_threshold]}")

plt.plot(list(scores.keys()), list(scores.values()))
plt.xlabel("Threshold")
plt.ylabel("Score")
