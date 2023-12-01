from notebooks.src.metric import score
import pandas as pd
import numpy as np

solution_df = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/solution_df.csv")
solution_df.loc[solution_df['id'].str.contains('kidney_3_sparse'), 'width'] = 1510
solution_df.loc[solution_df['id'].str.contains('kidney_3_sparse'), 'height'] = 1706
solution_df.loc[solution_df['id'].str.contains('kidney_3_sparse'), 'group'] = 'kidney_3_sparse'
solution_df.loc[solution_df['id'].str.contains('kidney_3_sparse'), 'slice'] = np.arange(len(solution_df))
model_dir = "efficientnet-b1_baseline_gray_scale_stack_images"
submission_df = pd.read_csv(f"/home/mithil/PycharmProjects/SenNetKideny/models/{model_dir}/oof.csv")
print(score(
    solution=solution_df,
    submission=submission_df,
    row_id_column_name='id',
    rle_column_name='rle',
    tolerance=0,
    image_id_column_name='group',
    slice_id_column_name='slice',
))

