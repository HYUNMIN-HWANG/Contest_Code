import pandas as pd
import random
import os
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils  import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

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

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

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

def modelfit(pip_xgb, grid_param_xgb, x, y) : 
    gs_xgb = (GridSearchCV(estimator=pip_xgb,
                        param_grid=grid_param_xgb,
                        cv=4,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=10))

    gs_xgb = gs_xgb.fit(x, y)
    print('Train Done.')

    #Predict training set:
    y_pred = gs_xgb.predict(x)
    #Print model report:
    print("\nModel Report")
    print("\nCV 결과 : ", gs_xgb.cv_results_)
    print("\n베스트 정답률 : ", gs_xgb.best_score_)
    print("\n베스트 파라미터 : ", gs_xgb.best_params_)


pip_xgb1 = Pipeline([('scl', StandardScaler()),
    ('reg', MultiOutputRegressor(xgb.XGBRegressor()))])
grid_param_xgb1 = {
    'reg__estimator__max_depth' : [5, 6, 7],
    'reg__estimator__gamma' : [1, 0.1, 0.01, 0.001, 0.0001, 0],
    'reg__estimator__learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.08],
    'reg__estimator__subsample' : [0.4, 0.6, 0.8],
    'reg__estimator__colsample_bytree' : [0.2, 0.6, 0.8]
}

modelfit(pip_xgb1, grid_param_xgb1, train_x, train_y)



# 베스트 정답률 :  -1.4531440581051838
# 베스트 파라미터 :  {'reg__estimator__colsample_bytree': 0.8, 'reg__estimator__gamma': 1, 'reg__estimator__learning_rate': 0.07, 'reg__estimator__max_depth': 7, 'reg__estimator__subsample': 0.8}