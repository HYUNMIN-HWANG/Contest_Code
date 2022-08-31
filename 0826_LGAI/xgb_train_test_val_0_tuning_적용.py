import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils  import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import sklearn.metrics as metrics

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'D:\\Data\\LGAI_AutoDriveSensors\\'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

# DATA
train_df = pd.read_csv(DATA_PATH + 'train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42, shuffle=True)
print(train_x.shape, val_x.shape)   # (31685, 56) (7922, 56)
print(train_y.shape, val_y.shape)   # (31685, 14) (7922, 14)

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt.iloc[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

# XGB Model Fit
xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, learning_rate=0.07, gamma = 1, subsample=0.8, colsample_bytree = 0.8, max_depth=7) ).fit(train_x, train_y,
                                                                                                                                                        verbose=1)
print('train Done.')

# validation set
val_pred = xgb.predict(val_x)
print('validation nrmse : ', lg_nrmse(val_y, val_pred))

# Inference
test_x = pd.read_csv(DATA_PATH + 'test.csv').drop(columns=['ID'])
preds = xgb.predict(test_x)
print("test Done")

# Submit
submit = pd.read_csv(DATA_PATH +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

print(submit.head())
submit.to_csv(DATA_PATH + 'submit/submit_train_test_val_0_tunning.csv', index=False)

# validation nrmse : 1.6386361359698076
# submit_train_test_val_0_tunning.csv
# score : 1.9533522069