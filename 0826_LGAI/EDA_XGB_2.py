from pyexpat.errors import XML_ERROR_UNCLOSED_CDATA_SECTION
import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils  import all_estimators
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler

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

def vectorization_pcb_2 (data) :
    label0 = data >= 0.0205+1.033e2
    label1 = data < 0.0205+1.033e2
    data[label0] = 0.0
    data[label1] = 1.0
    return data

def vectorization_pcb_5 (data) :
    label0 = data >= 102.5
    label1 = data < 102.5
    data[label0] = 0.0
    data[label1] = 1.0
    return data

def preprocessing_x (x_data) : 

    # PCB 체결 시 단계별 누름량 (2) vectorization
    pcb_3 = vectorization_pcb_2(x_data[['X_02']])

    # PCB 체결 시 단계별 누름량 (3) vectorization
    pcb_4 = vectorization_pcb_5(x_data[['X_05']])

    # 방열 재료 2,3 무게 펻균 칼럼 만들기
    Heating = group_mean(x_data, ['X_10','X_11'])
    x_data['X_57'] = Heating

    # PCB 체결 시 단계별 누름량 (1) (4) / 안테나 패드 위치 standardization / 스크류 체결 시 분당 회전수
    scaler = MinMaxScaler()
    # numerical_cols = ['X_01', 'X_06', 'X_14','X_15','X_16','X_17','X_18','X_34','X_35','X_36','X_37','X_24','X_25','X_26','X_27','X_28','X_29','X_50', 'X_51', 'X_52','X_53', 'X_54', 'X_55', 'X_56']
    numerical_cols = ['X_01', 'X_06', 'X_14','X_15','X_16','X_17','X_18','X_34','X_35','X_36','X_37','X_24','X_25','X_26','X_27','X_28','X_29','X_50', 'X_51', 'X_52', 'X_54', 'X_55', 'X_56', 'X_03','X_07','X_08','X_09','X_12','X_13','X_19','X_20','X_21','X_22','X_30','X_31','X_32','X_33','X_38','X_39','X_40','X_41','X_42','X_43','X_44','X_45','X_46','X_49']
    numerical_df = pd.DataFrame(scaler.fit_transform(x_data[numerical_cols]), columns=numerical_cols)
    
    # etc_column = ['X_03','X_07','X_08','X_09','X_12','X_13','X_19','X_20','X_21','X_22','X_30','X_31','X_32','X_33','X_38','X_39','X_40','X_41','X_42','X_43','X_44','X_45','X_46','X_49']

    x_new_data = pd.concat([pcb_3, pcb_4, x_data['X_57'], numerical_df], axis=1)
    return x_new_data


def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt.iloc[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

# XGB Model Fit
train_x = preprocessing_x(train_x)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma = 0, subsample=0.75, colsample_bytree = 1, max_depth=7)).fit(train_x, train_y)
print('train Done.')

# validation set
val_pred = xgb.predict(val_x)
print('validation nrmse : ', lg_nrmse(val_y, val_pred))

# Inference
test_x = pd.read_csv(DATA_PATH + 'test.csv').drop(columns=['ID'])
test_x = preprocessing_x(test_x)
preds = xgb.predict(test_x)
print("test Done")

# Submit
submit = pd.read_csv(DATA_PATH +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Submit Done.')

print(submit.head())
submit.to_csv(DATA_PATH + 'submit/EDA_XGB_2.csv', index=False)

# validation nrmse : 1.9470312982913134
# EDA_XGB_2.csv
# score : 2.2142072038	