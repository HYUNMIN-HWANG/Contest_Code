import pandas as pd
import random
import os
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ROOT_DIR = 'D:\\Data\\LGAI_AutoDriveSensors\\'
TEST_CSV = 'test.csv'
TRAIN_CSV = 'train.csv'
ORIGIN_SUBMIT_CSV = 'sample_submission.csv'

submit_now = datetime.datetime.now()
SUBMIT_CSV = 'submit/sample_submission_dateNtime_%d_%d_%d_%d_%d.csv' % (submit_now.month, submit_now.day, submit_now.hour, submit_now.minute, submit_now.second)

shutil.copy(os.path.join(ROOT_DIR, ORIGIN_SUBMIT_CSV), os.path.join(ROOT_DIR, SUBMIT_CSV))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv(os.path.join(ROOT_DIR, TRAIN_CSV))
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
    x_data.loc[:,'X_30'] = IQR_except_outlier(x_data.loc[:,'X_30'].copy())
    x_data.loc[:,'X_31'] = IQR_except_outlier(x_data.loc[:,'X_31'].copy())
    x_data.loc[:,'X_32'] = IQR_except_outlier(x_data.loc[:,'X_32'].copy())
    x_data.loc[:,'X_33'] = IQR_except_outlier(x_data.loc[:,'X_33'].copy())
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

train_x = preprocessing_x(train_x)

class LG_NRMSE(nn.Module):
  def __init__(self):
    super(LG_NRMSE, self).__init__()
    self.mse = nn.MSELoss().to(device)

  def forward(self, gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여

    all_nrmse_list = []
    # for idx in range(1,15): # ignore 'ID'
    for idx in range(1,14): # ignore ID
      rmse = torch.sqrt(self.mse(preds[:,idx], gt[:,idx]))
      nrmse = rmse / torch.mean(torch.abs(gt[:,idx]))
      all_nrmse_list.append(nrmse)
      all_nrmse = torch.stack(all_nrmse_list, dim=0)
    score = 1.2 * torch.sum(all_nrmse[:8]) + 1.0 * torch.sum(all_nrmse[8:15])
    return score

class Linear_net(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(Linear_net, self).__init__()
    self.dim_layer = [dim_in, 2*dim_in, 2*dim_out, dim_out]
    # self.dim_layer = [dim_in, 2*dim_in, 4*dim_in, 8*dim_in, 16*dim_in, 16*dim_out, 8*dim_out, 4*dim_out, 2*dim_out, dim_out]
    # self.dim_layer = [dim_in, 100, 200, 300, 400, 500, 400, 300, 200, 100, dim_out]
    self.layers = nn.ModuleList()
    for i in range(len(self.dim_layer) - 1):
      self.layers.append(nn.Sequential( # no ReLU, Dropout. These ruins the Result
          nn.Linear(self.dim_layer[i], self.dim_layer[i + 1]),
          nn.BatchNorm1d(self.dim_layer[i + 1])
      ))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

class Linear_net(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(Linear_net, self).__init__()
    self.dim_layer = [dim_in, 2*dim_in, 2*dim_out, dim_out]
    # self.dim_layer = [dim_in, 2*dim_in, 4*dim_in, 8*dim_in, 16*dim_in, 16*dim_out, 8*dim_out, 4*dim_out, 2*dim_out, dim_out]
    # self.dim_layer = [dim_in, 100, 200, 300, 400, 500, 400, 300, 200, 100, dim_out]
    self.layers = nn.ModuleList()
    for i in range(len(self.dim_layer) - 1):
      self.layers.append(nn.Sequential( # no ReLU, Dropout. These ruins the Result
          nn.Linear(self.dim_layer[i], self.dim_layer[i + 1]),
          nn.BatchNorm1d(self.dim_layer[i + 1])
      ))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    # for idx in range(1,15): # ignore 'ID'
    for idx in range(1,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:15])
    return score

lr = 1 # main
# wd = 0
wd = 10e-8 # sub main
batch_size = 1024
epochs = 1000
model = Linear_net(train_x.shape[1], train_y.shape[1]).to(device)
# print(train_x.shape[1], train_y.shape[1])

optimizer = optim.Adagrad(model.parameters(), lr = lr, weight_decay = wd)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 50, gamma=0.5)
# criterion = nn.MSELoss().to(device)
criterion = nn.L1Loss().to(device)
# criterion = LG_NRMSE().to(device)

# test set divisor
train_x = torch.from_numpy(train_x.to_numpy()).to(device).to(torch.float)
train_y = torch.from_numpy(train_y.to_numpy()).to(device).to(torch.float)

test_x_from_train = train_x[:train_x.shape[0] // 10]
test_y_from_train = train_y[:train_y.shape[0] // 10]
train_x = train_x[train_x.shape[0] // 10 :]
train_y = train_y[train_y.shape[0] // 10 :]

part_idx = len(train_x) // batch_size

print(train_x.shape[1], train_y.shape[1])

# train by k-fold
# best_score = 100
# for epoch in range(epochs):
#   running_loss = 0

#   for iter in range(part_idx + 1):
#     model.train()

#     val_x_part = train_x[iter * batch_size : (iter + 1) * batch_size]
#     val_y_part = train_y[iter * batch_size : (iter + 1) * batch_size]
#     train_x_part = torch.cat((
#         train_x[: iter * batch_size],
#         train_x[(iter + 1) * batch_size :]
#     ), 0)
#     train_y_part = torch.cat((
#         train_y[: iter * batch_size],
#         train_y[(iter + 1) * batch_size :]
#     ), 0)

#     if iter == part_idx:
#       val_x_part = train_x[iter * batch_size :]
#       val_y_part = train_y[iter * batch_size :]
#       train_x_part = train_x[: iter * batch_size]
#       train_y_part = train_y[: iter * batch_size]
      
#     pred_train = model(train_x_part)
#     loss = criterion(pred_train, train_y_part)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     with torch.no_grad():
#       model.eval()
#       pred_val = model(val_x_part)
#       val_loss = criterion(pred_val, val_y_part)
#       print("Epoch : %d, Iteration : %d, Validation Loss : %f"%(epoch + 1, iter + 1, val_loss))
      
#     running_loss += loss.item()

#   # scheduler.step()
#   print('--------------------------------------------')
#   print('--------------------------------------------')
#   print('--------------------------------------------')
#   print("Epoch : %d, Loss : %f"% (epoch + 1, running_loss / (part_idx)))
  
#   score = lg_nrmse(test_y_from_train.cpu().detach().numpy(), model(test_x_from_train).cpu().detach().numpy())

#   if best_score > score :
#     # 추가 정보
#     now = datetime.datetime.now()
#     EPOCH = epoch
#     PATH = ROOT_DIR + "weights/model3_epoch_%d_dateNtime_%d_%d_%d_%d_%d" % (epoch + 1, now.month, now.day, now.hour, now.minute, now.second)
#     LOSS = running_loss

#     torch.save({
#                 'epoch': EPOCH,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': LOSS,
#                 }, PATH)
#     best_score = score


test_x = pd.read_csv(os.path.join(ROOT_DIR, TEST_CSV)).drop(columns=['ID'])
test_x = preprocessing_x(test_x)
test_x = torch.from_numpy(test_x.to_numpy()).to(device).to(torch.float)

PATH = ROOT_DIR + 'weights/model3_epoch_9_dateNtime_8_15_13_10_4'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

with torch.no_grad():
  model.eval()
  preds = model(test_x)
  preds = preds.cpu().detach().numpy()

print('Done.')

submit = pd.read_csv(os.path.join(ROOT_DIR, SUBMIT_CSV))
for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')
submit.to_csv(os.path.join(ROOT_DIR, SUBMIT_CSV), index=False)


# TEST
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

test_model = Linear_net(train_x.shape[1], train_y.shape[1]).to(device)
optimizer = optim.Adagrad(test_model.parameters(), lr = lr, weight_decay = wd)

############################################################ fix this path
PATH = ROOT_DIR + 'weights/model3_epoch_9_dateNtime_8_15_13_10_4'

checkpoint = torch.load(PATH)
test_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

test_model.eval()
print(" ")
# test_model.train()

score = lg_nrmse(test_y_from_train.cpu().detach().numpy(), test_model(test_x_from_train).cpu().detach().numpy())
print(score) # 1.8171342343091965

# sample_submission_dateNtime_8_15_13_13_32.csv
# 