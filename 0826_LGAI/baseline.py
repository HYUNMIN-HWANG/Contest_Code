import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

# DATA
train_df = pd.read_csv('D:\\Data\\LGAI_AutoDriveSensors\\train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

# Regression Model Fit
LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
print('Done.')

# Inference
test_x = pd.read_csv('D:\\Data\\LGAI_AutoDriveSensors\\test.csv').drop(columns=['ID'])

preds = LR.predict(test_x)
print('Done.')

# Submit
submit = pd.read_csv('D:\\Data\\LGAI_AutoDriveSensors\\sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('D:\\Data\\LGAI_AutoDriveSensors\\submit\\baseline.csv', index=False)