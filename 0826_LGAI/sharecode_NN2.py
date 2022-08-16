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
#     PATH = ROOT_DIR + "weights/model_epoch_%d_dateNtime_%d_%d_%d_%d_%d" % (epoch + 1, now.month, now.day, now.hour, now.minute, now.second)
#     LOSS = running_loss

#     torch.save({
#                 'epoch': EPOCH,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': LOSS,
#                 }, PATH)
#     best_score = score

test_x = pd.read_csv(os.path.join(ROOT_DIR, TEST_CSV)).drop(columns=['ID'])
test_x = torch.from_numpy(test_x.to_numpy()).to(device).to(torch.float)

PATH = ROOT_DIR + 'weights/model_epoch_747_dateNtime_8_14_19_1_8'
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
PATH = ROOT_DIR + 'weights/model_epoch_747_dateNtime_8_14_19_1_8'

checkpoint = torch.load(PATH)
test_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

test_model.eval()
print(" ")
# test_model.train()

score = lg_nrmse(test_y_from_train.cpu().detach().numpy(), test_model(test_x_from_train).cpu().detach().numpy())
print(score) # 1.824580931663513

# sample_submission_dateNtime_8_15_12_36_13.csv
# 2.0004388766