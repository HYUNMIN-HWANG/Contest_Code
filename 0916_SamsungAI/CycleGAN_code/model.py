from torch.nn.modules import normalization, padding
from layer import *

import torch
import torch.nn as nn
from torch.nn import init

# https://github.com/hanyoseob/youtube-cnn-007-pytorch-cyclegan/blob/master/model.py

# CycleGAN
class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="inorm"):
        super(CycleGAN, self).__init__()
        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=7, padding=(7-1)//2, norm=norm, relu=0.0, stride=1)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)

        res = []
        for i in range(9) :
            res += [ResBlock(4 * nker, 4 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=1)]
        self.res = nn.Sequential(*res)

        self.dec3 = DECBR2d(4 * nker, 2 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.dec2 = DECBR2d(2 * nker, 1 * nker, kernel_size=3, padding=(3-1)//2, norm=norm, relu=0.0, stride=2)
        self.dec1 = CBR2d(1 * nker, out_channels, kernel_size=7, padding=(7-1)//2, norm=None, relu=None, stride=1)

    def forward(self, x) :
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.res(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x

class Pix2Pix(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Pix2Pix, self).__init__()

        # Encoder
        self.enc1 = CBR2d(in_channels, 1 * nker, kernel_size=4, padding=1, norm=None, relu=0.2, stride=2)
        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc5 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc6 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc7 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)
        self.enc8 = CBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.2, stride=2)

        # Decoder
        self.dec1 = DECBR2d(8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.drop1 = nn.Dropout2d(0.5)

        self.dec2 = DECBR2d(2 * 8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)    # 나중에 encoder와 concat하기 때문에 인풋 사이즈가 2배 늘어남
        self.drop2 = nn.Dropout2d(0.5)

        self.dec3 = DECBR2d(2 * 8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.drop3 = nn.Dropout2d(0.5)

        self.dec4 = DECBR2d(2 * 8 * nker, 8 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec5 = DECBR2d(2 * 8 * nker, 4 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec6 = DECBR2d(2 * 4 * nker, 2 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec7 = DECBR2d(2 * 2 * nker, 1 * nker, kernel_size=4, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec8 = DECBR2d(2 * 1 * nker, out_channels, kernel_size=4, padding=1, norm=None, relu=None, stride=2)

    def forward (self, x) :
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec1 = self.dec1(enc8)
        drop1 = self.drop1(dec1)

        cat2 = torch.cat((drop1, enc7), dim=1)
        dec2 = self.dec2(cat2)
        drop2 = self.drop2(dec2)

        cat3 = torch.cat((drop2, enc6), dim=1)
        dec3 = self.dec3(cat3)
        drop3 = self.drop3(dec3)

        cat4 = torch.cat((drop3, enc5), dim=1)
        dec4 = self.dec4(cat4)

        cat5 = torch.cat((dec4, enc4), dim=1)
        dec5 = self.dec5(cat5)

        cat6 = torch.cat((dec5, enc3), dim=1)
        dec6 = self.dec6(cat6)

        cat7 = torch.cat((dec6, enc2), dim=1)
        dec7 = self.dec7(cat7)

        cat8 = torch.cat((dec7, enc1), dim=1)
        dec8 = self.dec8(cat8)

        x = torch.tanh(dec8)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm"):
        super(UNet, self).__init__()

        self.learning_type = learning_type

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=in_channels, out_channels=1 * nker, norm=norm)
        self.enc1_2 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=nker, out_channels=2 * nker, norm=norm)
        self.enc2_2 = CBR2d(in_channels=2 * nker, out_channels=2 * nker, norm=norm)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=2 * nker, out_channels=4 * nker, norm=norm)
        self.enc3_2 = CBR2d(in_channels=4 * nker, out_channels=4 * nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=4 * nker, out_channels=8 * nker, norm=norm)
        self.enc4_2 = CBR2d(in_channels=8 * nker, out_channels=8 * nker, norm=norm)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=8 * nker, out_channels=16 * nker, norm=norm)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=16 * nker, out_channels=8 * nker, norm=norm)

        self.unpool4 = nn.ConvTranspose2d(in_channels=8 * nker, out_channels=8 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 8 * nker, out_channels=8 * nker, norm=norm)
        self.dec4_1 = CBR2d(in_channels=8 * nker, out_channels=4 * nker, norm=norm)

        self.unpool3 = nn.ConvTranspose2d(in_channels=4 * nker, out_channels=4 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 4 * nker, out_channels=4 * nker, norm=norm)
        self.dec3_1 = CBR2d(in_channels=4 * nker, out_channels=2 * nker, norm=norm)

        self.unpool2 = nn.ConvTranspose2d(in_channels=2 * nker, out_channels=2 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 2 * nker, out_channels=2 * nker, norm=norm)
        self.dec2_1 = CBR2d(in_channels=2 * nker, out_channels=1 * nker, norm=norm)

        self.unpool1 = nn.ConvTranspose2d(in_channels=1 * nker, out_channels=1 * nker,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 1 * nker, out_channels=1 * nker, norm=norm)
        self.dec1_1 = CBR2d(in_channels=1 * nker, out_channels=1 * nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=1 * nker, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = x + self.fc(dec1_1)

        return x
    

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, learning_type="plain", norm="bnorm", nblk=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = CBR2d(in_channels, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=None, relu=0.0)

        res = []
        for i in range(nblk):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)]
        self.res = nn.Sequential(*res)

        self.dec = CBR2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=0.0)

        self.fc = CBR2d(nker, out_channels, kernel_size=1, stride=1, padding=0, bias=True, norm=None, relu=None)

    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)
        elif self.learning_type == "residual":
            x = x0 + self.fc(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, out_channels, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x

# DCGAN
class DCGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN, self).__init__()

        self.dec1 = DECBR2d(1 * in_channels, 8 * nker, kernel_size=4, stride=1,
                            padding=0, norm=norm, relu=0.0, bias=False)

        self.dec2 = DECBR2d(8 * nker, 4 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec3 = DECBR2d(4 * nker, 2 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec4 = DECBR2d(2 * nker, 1 * nker, kernel_size=4, stride=2,
                            padding=1, norm=norm, relu=0.0, bias=False)

        self.dec5 = DECBR2d(1 * nker, out_channels, kernel_size=4, stride=2,
                            padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        x = torch.tanh(x)

        return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__    # 클래스 이름 참조하기
        # hasattr : argument로 넘겨준 object 에 name 의 속성이 존재하면 True, 아니면 False를 반환
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1) :
            if init_type == 'normal' :
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming' :
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else :
                # "아직 구현하지 않은 부분입니다" 라는 오류를 강제로 발생
                raise NotImplemented('initialization method [%s] is not implemented' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net