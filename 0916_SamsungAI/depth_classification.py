import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import pandas as pd

train_dir = 'D://Data//3D_Metrology//train'

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect

def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
seed_everything(42) # Seed 고정

transform_train = transforms.Compose([#Resize(shape=(286, 286, nch)),
                                        #RandomCrop((ny, nx)),                 # Random Jitter
                                        #Normalization(mean=0.5, std=0.5)])
                                    ])    

sem_meta = pd.read_csv(train_dir+"\\real_sim_sem_meta.csv")
sem_val = sem_meta.sample(n=int(sem_meta.shape[0]*0.1), random_state=42).dropna().reset_index()
sem_train = sem_meta.drop(sem_val.index).dropna().reset_index()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.to_tensor = ToTensor()

        self.lst_data_real = data_dir['REAL'].dropna()   # A

    def __getitem__(self, index):
        # import ipdb; ipdb.set_trace()
        # X
        data_a = Image.open(self.lst_data_real[index]).convert('L')
        data_a = np.array(data_a)

        if data_a.ndim == 2:
            data_a = data_a[:, :, np.newaxis]
        if data_a.dtype == np.uint8:
            data_a = data_a / 255.0

        if self.transform:
            data_a = self.transform(data_a)

        data_a = data_a.transpose((2, 0, 1)).astype(np.float32)
        X = torch.from_numpy(data_a)

        # y
        depth_class = os.path.dirname((os.path.dirname(self.lst_data_real[index])))[-3:]

        if depth_class == '110' :
            y = 0
        elif depth_class == '120' :
            y = 1
        elif depth_class == '130' :
            y = 2
        else :
            y = 3

        return X, y
    
    def __len__(self):
        return len(self.lst_data_real)

class ToTensor(object):
    def __call__(self, data) :
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

batch_size = 32

dataset_train = Dataset(data_dir=sem_train,
                        transform=transform_train)

loader_train = DataLoader(dataset_train, batch_size=batch_size,
                            shuffle=True, num_workers=0)

num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)
print(num_data_train)   # 54611

dataset_val = Dataset(data_dir=sem_val,
                        transform=transform_train)

loader_val = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=True, num_workers=0)

num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)
print(num_data_val)   # 6053

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# NETWORK
EPOCH = 300
in_channels = 1

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

best_AUROC = 0

# TRAIN
lowest_loss = 100
for epoch in range(0, EPOCH):
    model = model.to(device)
    model.train()

    # import ipdb; ipdb.set_trace()
    for i, (X, y) in enumerate(loader_train):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)

        prediction = model.forward(X)
        
        loss = criterion(prediction.softmax(dim=1), y)
        loss.backward()
        optimizer.step()

        if i % 100 == 0 :
            print("Train epoch {} [batch {}/{}] loss {:.4f}".format(epoch, i, len(loader_train), loss))

    # EVALUATION
    val_losses = 0

    with torch.no_grad():  
        for x_val, y_val in loader_val:  
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            model.eval()  

            prediction = model.forward(x_val)  

            val_losses += criterion(prediction.softmax(dim=1), y_val)

    print(f"Epoch {epoch+1}/{EPOCH}, validation loss: {val_losses/len(loader_val)}")   

    if lowest_loss > (val_losses/len(loader_val)) :
        torch.save({'model_state_dict' : model.state_dict()},'D:\\Data\\3D_Metrology\\classification_weight/resnet50/model_'+str(epoch)+'.pth')
        lowest_loss = (val_losses/len(loader_val))
