import os
import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
from utils import *
from PIL import Image

# https://github.com/hanyoseob/youtube-cnn-007-pytorch-cyclegan/blob/master/dataset.py

class Dataset_origin(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, data_type='both'):
        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'
        self.transform = transform
        self.task = task
        self.data_type = data_type

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        if os.path.exists(self.data_dir_a):
            lst_data_a = os.listdir(self.data_dir_a)
            lst_data_a = [f for f in lst_data_a if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_data_a.sort()
        else:
            lst_data_a = []

        if os.path.exists(self.data_dir_b):
            lst_data_b = os.listdir(self.data_dir_b)
            lst_data_b = [f for f in lst_data_b if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
            lst_data_b.sort()
        else:
            lst_data_b = []

        self.lst_data_a = lst_data_a
        self.lst_data_b = lst_data_b

    def __getitem__(self, index):
        
        data = {}
        if self.data_type == 'a' or self.data_type == 'both':
            data_a = plt.imread(os.path.join(self.data_dir_a, self.lst_data_a[index]))[:, :, :3]

            if data_a.ndim == 2:
                data_a = data_a[:, :, np.newaxis]
            if data_a.dtype == np.uint8:
                data_a = data_a / 255.0

            data['data_a'] = data_a

        if self.data_type == 'b' or self.data_type == 'both':
            data_b = plt.imread(os.path.join(self.data_dir_b, self.lst_data_b[index]))[:, :, :3]

            if data_b.ndim == 2:
                data_b = data_b[:, :, np.newaxis]
            if data_b.dtype == np.uint8:
                data_b = data_b / 255.0

            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data
    
    def __len__(self):
        # 인덱스 길이가 짧은 쪽에 맞춰서 길이를 정의해야 한다.
        if self.data_type == 'both' :   # a와 b 모두 출력
            if len(self.lst_data_a) < len(self.lst_data_b):
                return len(self.lst_data_a)
            else :
                return len(self.lst_data_b)
        elif self.data_type == 'a' :    # a만 출력
            return len(self.lst_data_a)
        elif self.data_type == 'b' :    # b만 출력
            return len(self.lst_data_b)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, data_type='both'):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.data_type = data_type

        self.to_tensor = ToTensor()

        if data_type=='both' :
            self.lst_data_real = data_dir['REAL'].dropna()   # A
            self.lst_data_sim = data_dir['SIM']    # B
        elif data_type=='b' :
            self.lst_data_sim = data_dir['SIM']    # B
        else :
            self.lst_data_real = data_dir['REAL'].dropna()   # A

    def __getitem__(self, index):
        
        data = {}
        if self.data_type == 'a' or self.data_type == 'both':
            # data_a = plt.imread(self.lst_data_real[index])
            data_a = Image.open(self.lst_data_real[index]).convert('L')
            data_a = np.array(data_a)

            if data_a.ndim == 2:
                data_a = data_a[:, :, np.newaxis]
            if data_a.dtype == np.uint8:
                data_a = data_a / 255.0

            data['data_a'] = data_a

        if self.data_type == 'b' or self.data_type == 'both':
            data_b = Image.open(self.lst_data_sim[index])
            data_b = np.array(data_b)

            if data_b.ndim == 2:
                data_b = data_b[:, :, np.newaxis]
            if data_b.dtype == np.uint8:
                data_b = data_b / 255.0

            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data
    
    def __len__(self):
        # 인덱스 길이가 짧은 쪽에 맞춰서 길이를 정의해야 한다.
        if self.data_type == 'both' :   # a와 b 모두 출력
            if len(self.lst_data_real) < len(self.lst_data_sim):
                return len(self.lst_data_real)
            else :
                return len(self.lst_data_sim)
        elif self.data_type == 'a' :    # a만 출력
            return len(self.lst_data_real)
        elif self.data_type == 'b' :    # b만 출력
            return len(self.lst_data_sim)

class Dataset_SIM(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, data_type='both'):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.data_type = data_type

        self.to_tensor = ToTensor()

        if data_type=='both' :
            self.lst_data_real = data_dir['SEM'].dropna()   # A
            self.lst_data_sim = data_dir['Depth']    # B
        elif data_type=='b' :
            self.lst_data_sim = data_dir['Depth']    # B
        else :
            self.lst_data_real = data_dir['SEM'].dropna()   # A

    def __getitem__(self, index):
        
        data = {}
        if self.data_type == 'a' or self.data_type == 'both':
            data_a = Image.open(self.lst_data_real[index]).convert('L')
            data_a = np.array(data_a)

            if data_a.ndim == 2:
                data_a = data_a[:, :, np.newaxis]
            if data_a.dtype == np.uint8:
                data_a = data_a / 255.0

            data['data_a'] = data_a

        if self.data_type == 'b' or self.data_type == 'both':
            data_b = Image.open(self.lst_data_sim[index])
            data_b = np.array(data_b)

            if data_b.ndim == 2:
                data_b = data_b[:, :, np.newaxis]
            if data_b.dtype == np.uint8:
                data_b = data_b / 255.0

            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data
    
    def __len__(self):
        # 인덱스 길이가 짧은 쪽에 맞춰서 길이를 정의해야 한다.
        if self.data_type == 'both' :   # a와 b 모두 출력
            if len(self.lst_data_real) < len(self.lst_data_sim):
                return len(self.lst_data_real)
            else :
                return len(self.lst_data_sim)
        elif self.data_type == 'a' :    # a만 출력
            return len(self.lst_data_real)
        elif self.data_type == 'b' :    # b만 출력
            return len(self.lst_data_sim)

class ToTensor(object):
    def __call__(self, data) :
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class Normalization(object) :
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std
        
        return data


class RandomFlip(object) :
    def __call__(self, data) :
        if np.random.rand() > 0.5:
            data = np.fliplr(data)                  # np.fliplr : Reverse the order of elements along axis 1

            for key, value in data.items() :
                data[key] = np.flip(value, axis=0)  # np.flip : Reverse the order of elements in an array along the given axis.

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)
            
        return data

class Rescale(object) :
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))    # assert : 어떤 조건이 True임을 보증하기 위한 것(True면 그대로 진행한다) # isinstance 함수 : 주어진 인스턴스가 특정 클래스/데이터 타입인지 검사해주는 함수
        self.output_size = output_size
    
    def __call__(self, data):
        h, w = data.shape[:2]

        if isinstance(self.output_size, int) :
            if h > w :
                new_h, new_w = self.output_size * h / w, self.output_size
            else :
                new_h, new_w = self.output_size, self.output_size * w / h
        else :
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))
        return data

class CenterCrop(object):      
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int) :
            self.output_size = (output_size, output_size)
        else :
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, data):
        h, w = data.shape[:2]

        new_h, new_w = self.output_size

        top = int(abs(h - new_h) / 2)
        left = int(abs(w - new_w) / 2)

        data = data[top: top + new_h, left: left + new_w]

        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):

        key = list(data.keys())[0]  # 첫 번째 key값으로 h,w를 구한다.

        h, w = data[key].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data

class ToNumpy(object):
    def __call__(self, data):
        if data.ndim == 3:
            data = data.to('cpu').detach().numpy().transpose((1,2,0))
        elif data.ndim == 4 :
            data = data.to('cpu').detach().numpy().transpose((0, 2, 3, 1))
        
        return data
        
class Denomalize(object):
    def __call__(self, data):

        return (data + 1) / 2

class Resize(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data) :
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))
        
        return data