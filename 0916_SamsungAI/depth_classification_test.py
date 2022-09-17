import os
import numpy as np
import random
from PIL import Image
from torchvision import transforms
import pandas as pd

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

test_dir = 'D://Data//3D_Metrology//test//SEM'
transform_test = transforms.Compose([#Resize(shape=(286, 286, nch)),
                                        #RandomCrop((ny, nx)),                 # Random Jitter
                                        #Normalization(mean=0.5, std=0.5)])
                                    ])    

# sem_meta = pd.read_csv(test_dir+"\\test_sem_meta_class.csv")
sem_meta = pd.read_csv(test_dir+"\\test_sem_meta_class_resnet50.csv")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

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

        return X
    
    def __len__(self):
        return len(self.lst_data_real)

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

batch_size = 32

dataset_test = Dataset(data_dir=sem_meta,
                        transform=transform_test)

loader_test = DataLoader(dataset_test, batch_size=batch_size,
                            shuffle=False, num_workers=0)

num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)
print(num_batch_test)   # 54611

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# NETWORK
EPOCH = 300
in_channels = 1

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

load_model = torch.load("D:\\Data\\3D_Metrology\\classification_weight\\resnet50\\model_298.pth", map_location=device)
model.load_state_dict(load_model['model_state_dict'])

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

best_AUROC = 0

# TEST
lowest_loss = 100
CLASS_list = []
with torch.no_grad():  
    # import ipdb; ipdb.set_trace()
    for x_val in loader_test:  
        model.eval()  

        prediction = model.forward(x_val)  
        # print(prediction)
        prediction = prediction.softmax(dim=1)
        prediction_class = np.argmax(prediction, axis=1)
        CLASS_list.extend(prediction_class)
        print(prediction_class)


print(len(CLASS_list))
CLASS_df = pd.DataFrame(CLASS_list)
sem_meta['Class'] = CLASS_df
print(sem_meta.head())
print(sem_meta.tail())

# sem_meta.to_csv(test_dir+"\\test_sem_meta_class.csv", index=False)
sem_meta.to_csv(test_dir+"\\test_sem_meta_class_resnet50.csv", index=False)
