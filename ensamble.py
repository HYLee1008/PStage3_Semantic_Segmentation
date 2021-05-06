import pandas as pd
import numpy as np


def hard_voting(target):
    N, B = target.shape[0], target.shape[1]
    result = np.zeros_like(target[0])
    for i in range(B):
        result[i] = np.bincount(target[:, i]).argmax()

    return result


files_list = [
    './ensamble/deeplabv3_resnet101 StepLR.csv', 
    './ensamble/DeepLabV3Plus_dpn92_2_mIoU_TTA_average.csv', 
    './ensamble/DeepLABV3plus_resnet101_2021_05_06_3.csv',
    # './ensamble/DeepLabV3Plus_dpn92_2_mIoU_TTA_hardvote.csv'
    ]
df_list = [0] * len(files_list)

# file = './submission/DeepLabV3Plus_dpn92_2_mIoU_TTA_average.csv'

# df = pd.read_csv(file, index_col=None)

for i, file in enumerate(files_list):
    df_list[i] = pd.read_csv(file, index_col=None)


final_result = np.zeros((len(df_list[0]), 65536), dtype=np.int32)
for i in range(len(df_list[0])):
    predictions = np.zeros((len(files_list),65536), dtype=np.int32)
    for j in range(len(files_list)):
        predictions[j] = np.fromstring(df_list[j]['PredictionString'][i], dtype=np.int32, sep=' ')

    final_result[i] = hard_voting(predictions)

final_result = np.array(final_result, dtype=np.int32)

submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

# PredictionString 대입
for file_name, string in zip(df_list[0]['image_id'], final_result):
    submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                ignore_index=True)

# submission.csv로 저장
submission.to_csv(f"./ensamble/final_submission.csv", index=False)