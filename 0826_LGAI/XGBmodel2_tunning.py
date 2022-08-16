from tkinter import N, Grid
import pandas as pd
import random
import os
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

# DATA Preprocessing
def drop_columns(x, cols) : 
    return x.drop(cols, axis='columns')

def group_mean(x, cols) :
    x_cols = x.loc[:,cols]
    cols_means = x_cols.mean(axis='columns')
    return cols_means

def add_col(x, col_name, value) :
    x[col_name] = value

def make_mean_col(x, cols, col_name) :
    cols_mean = group_mean(x, cols)
    x = drop_columns(x, cols)
    add_col(x, col_name, cols_mean)
    return x

def IQR_except_outlier(col) : 
    Q3, Q1 = np.percentile(col, [75, 25])
    IQR = Q3 - Q1
    lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
    data_low_idx = col[lower > col].index
    data_upper_idx = col[upper < col].index
    execpt_outlier = col[(lower < col) & (upper > col)]

    # outlier를 뺀 값들의 min과 max 값으로 대체함
    min = execpt_outlier.min()
    max = execpt_outlier.max()

    col[data_low_idx] = min
    col[data_upper_idx] = max

    return col

def standardization (data) :
    # z = (x - mean())/std()
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def preprocessing_x (x_data) : 
    # 1차, 2차, 3차 4차 검사 통과 여부 칼럼 제거함 (모두 동일한 값 1, 모두 통과함)
    pass_columns = ['X_04','X_23','X_47','X_48']
    x_data = drop_columns(x_data, pass_columns)

    # 안테나 패드 위치 평균 칼럼 만들기 'X_57'
    x_data = make_mean_col(x_data, ['X_14','X_15','X_16','X_17','X_18'], 'X_57')
    # n번 스크류 삽입 깊이 평균 칼럼 만들기 'X_58'
    x_data = make_mean_col(x_data, ['X_19','X_20','X_21','X_22'], 'X_58')
    # 커넥터 핀 치수 평균 칼럼 만들기 'X_59'
    x_data = make_mean_col(x_data, ['X_24','X_25','X_26','X_27','X_28','X_29'], 'X_59')
    # 스크류 삽입 깊이 n 평균 칼럼 만들기 'X_60'
    x_data = make_mean_col(x_data, ['X_30','X_31','X_32','X_33'], 'X_60')
    # 스크류 체결 시 분당 회전 수 평균 칼럼 만들기 'X_61'
    x_data = make_mean_col(x_data, ['X_34','X_35','X_36','X_37'], 'X_61')
    # 하우징 PCB 안착부 평균 칼럼 만들기 'X_62'
    x_data = make_mean_col(x_data, ['X_38','X_39','X_40'], 'X_62')
    # 레이돔 치수 평균 칼럼 만들기 'X_63'
    x_data = make_mean_col(x_data, ['X_41','X_42','X_43','X_44'], 'X_63')
    # RF 부붙 SMT 납 량 평균 칼럼 만들기 'X_64'
    x_data = make_mean_col(x_data, ['X_50','X_51','X_52','X_53','X_54','X_55','X_56'], 'X_64')
    # 방열 재료 2,3 무게 평균 칼럼 만들기 'X_65'
    x_data = make_mean_col(x_data, ['X_10','X_11'], 'X_65')

    # IQR 사용하여 이상치 처리, 이상치를 제거했을 때의 min과 max 값으로 대체
    x_data.loc[:,'X_06'] = IQR_except_outlier(x_data.loc[:,'X_06'].copy())
    x_data.loc[:,'X_07'] = IQR_except_outlier(x_data.loc[:,'X_07'].copy())
    x_data.loc[:,'X_08'] = IQR_except_outlier(x_data.loc[:,'X_08'].copy())
    x_data.loc[:,'X_09'] = IQR_except_outlier(x_data.loc[:,'X_09'].copy())
    x_data.loc[:,'X_49'] = IQR_except_outlier(x_data.loc[:,'X_49'].copy())

    # 표준화
    x_data = standardization(x_data)
    # >> train_x.shape (39607, 22)
    return x_data

train_x_stand = preprocessing_x(train_x)

# XGB Model Fit

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


# pip_xgb1 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor()))])
# grid_param_xgb1 = {
#     'reg__estimator__max_depth' : [3, 6, 9],
#     'reg__estimator__min_child_weight' : [1, 3, 5]
# }
# modelfit(pip_xgb1, grid_param_xgb1, train_x_stand, train_y)
# 베스트 파라미터 : {'reg__estimator__max_depth': 3, 'reg__estimator__min_child_weight': 3}

# pip_xgb2 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3)))])
# grid_param_xgb2 = {
#     'reg__estimator__gamma' : [i/10.0 for i in range(0,5)]
# }
# modelfit(pip_xgb2, grid_param_xgb2, train_x_stand, train_y)
# 베스트 파라미터 :  {'reg__estimator__gamma': 0.1}

# pip_xgb3 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3, gamma=0.1)))])
# grid_param_xgb3 = {
#     'reg__estimator__subsample' : [i/10.0 for i in range(6,10)],
#     'reg__estimator__colsample_bytree' : [i/10.0 for i in range(6,10)]
# }
# modelfit(pip_xgb3, grid_param_xgb3, train_x_stand, train_y)
# 베스트 파라미터 :  {'reg__estimator__colsample_bytree': 0.7, 'reg__estimator__subsample': 0.9}

# pip_xgb4 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3, gamma=0.1, 
#                                                   subsample=0.9, colsample_bytree=0.7)))])
# grid_param_xgb4 = {
#     'reg__estimator__subsample' : [i/100.0 for i in range(40,80)]
# }
# modelfit(pip_xgb4, grid_param_xgb4, train_x_stand, train_y)
# 베스트 파라미터 :  {'reg__estimator__subsample': 0.79}

# pip_xgb5 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3, gamma=0.1, 
#                                                   subsample=0.79, colsample_bytree=0.7)))])
# grid_param_xgb5 = {
#     'reg__estimator__reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100]
# }
# modelfit(pip_xgb5, grid_param_xgb5, train_x_stand, train_y)
# 베스트 파라미터 :  {'reg__estimator__reg_alpha': 100}

# pip_xgb6 = Pipeline([('scl', StandardScaler()),
#     ('reg', MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3, gamma=0.1, 
#                                                   subsample=0.79, colsample_bytree=0.7, reg_alpha=100)))])
# grid_param_xgb6 = {
#     'reg__estimator__laerning_rate' : [0.01, 0.05, 0.1, 0.2]
# }
# modelfit(pip_xgb6, grid_param_xgb6, train_x_stand, train_y)
# 베스트 파라미터 :  {'reg__estimator__laerning_rate': 0.01}

# MultiOutputRegressor(xgb.XGBRegressor(max_depth=3, min_child_weight=3, gamma=0.1, 
#                                                   subsample=0.79, colsample_bytree=0.7, reg_alpha=100, learning_rate=0.01))