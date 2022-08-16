import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt

import xgboost as xgb

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

# XGB Model Fit
xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y)
print('Done.')

# Inference
test_x = pd.read_csv(DATA_PATH + 'test.csv').drop(columns=['ID'])

preds = xgb.predict(test_x)
print('Done.')

# Submit
submit = pd.read_csv(DATA_PATH +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv(DATA_PATH + 'submit/submit_xgb.csv', index=False)