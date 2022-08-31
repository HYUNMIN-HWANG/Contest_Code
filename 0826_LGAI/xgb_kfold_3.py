from sys import set_asyncgen_hooks
from tabnanny import verbose
import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils  import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

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

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

test_x = pd.read_csv(DATA_PATH + 'test.csv').drop(columns=['ID'])
train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_x = test_x.to_numpy()

print(train_x.shape, train_y.shape)  # (39607, 56) (39607, 14)
print(test_x.shape)                  # (39608, 56)

N_ESTIMATORS = 1000
N_SPLITS = 10
SEED = 42
EARLY_STOPPING_ROUNDS = 20
VERBOSE = 0

skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
y_pred = np.zeros([test_x.shape[0], train_y.shape[1]])


#KFold
for fold, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)) :
    # import ipdb; ipdb.set_trace()
    print(f"=====Fold {fold}=====")

    x_train_sub = train_x[train_idx]
    x_val_sub = train_x[valid_idx]

    y_train_sub = train_y[train_idx]
    y_val_sub = train_y[valid_idx]  

    eval_set = [(x_val_sub, y_val_sub)]
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=200, learning_rate=0.07, gamma = 1, subsample=0.8, colsample_bytree = 0.8, max_depth=7, eval_set=eval_set) )
    model.fit(x_train_sub, y_train_sub, verbose=True)

    y_val_pred = model.predict(x_val_sub)
    nrmse_score = lg_nrmse(y_val_sub, y_val_pred)
    print(f"===== lg_nrmse {nrmse_score:.6f} =====")    

    y_pred += model.predict(test_x)

nrmse_score = lg_nrmse(y_val_sub, y_val_pred)
print(f"===== lg_nrmse {nrmse_score:.6f} =====")    

y_pred /= N_SPLITS
print(y_pred.shape)

# Submit
submit = pd.read_csv(DATA_PATH +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = y_pred[:,idx-1]
print('Done.')

print(submit.head())
submit.to_csv(DATA_PATH + 'submit/kfold_3.csv', index=False)

# ===== lg_nrmse 1.635527 =====
# kfold_3.csv
# score : overfitting