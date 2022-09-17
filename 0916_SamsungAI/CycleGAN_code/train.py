import os
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import itertools

from model import *
from dataset import *
from utils import *

import matplotlib.pyplot as plt
import zipfile

from torchvision import transforms
# https://github.com/hanyoseob/youtube-cnn-007-pytorch-cyclegan/blob/master/train.py
def train(args) :
    mode = args.mode
    train_continue = args.train_continue
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)
    print("norm: %s" % norm)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
    seed_everything(42) # Seed 고정

    ## setup dataset
    result_dir_train = os.path.join(result_dir, 'train')

    if not os.path.exists(result_dir_train) :
        os.makedirs(os.path.join(result_dir_train, 'png'))


    ## Network Training
    if mode == 'train' :
        transform_train = transforms.Compose([#Resize(shape=(286, 286, nch)),
                                              #RandomCrop((ny, nx)),                 # Random Jitter
                                              #Normalization(mean=0.5, std=0.5)])
                                            ])    
        
        sem_meta = pd.read_csv(data_dir+"\\real_sim_sem_meta.csv")
        # sim_val = sem_meta.sample(n=int(sem_meta.shape[0]*0.1), random_state=42)
        # sim_train = sem_meta.drop(sim_val.index)
        
        dataset_train = Dataset(data_dir=sem_meta,
                                transform=transform_train, task=task,
                                data_type='both')

        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    elif mode == 'train_SIM' :
        transform_train = transforms.Compose([#Resize(shape=(286, 286, nch)),
                                              #RandomCrop((ny, nx)),                 # Random Jitter
                                              #Normalization(mean=0.5, std=0.5)])
                                            ])    
        
        sem_meta = pd.read_csv(data_dir+"\\simulation_meta_3.csv")
        # sim_val = sem_meta.sample(n=int(sem_meta.shape[0]*0.1), random_state=42)
        # sim_train = sem_meta.drop(sim_val.index)
        
        dataset_train = Dataset_SIM(data_dir=sem_meta,
                                transform=transform_train, task=task,
                                data_type='both')

        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    elif mode == "train_sem_depth" :
        transform_train = transforms.Compose([#Resize(shape=(286, 286, nch)),
                                              #RandomCrop((ny, nx)),                 # Random Jitter
                                              #Normalization(mean=0.5, std=0.5)])
                                            ])    
        
        sem_meta = pd.read_csv(data_dir+"\\real_sem_depth_meta.csv")
        # sim_val = sem_meta.sample(n=int(sem_meta.shape[0]*0.1), random_state=42)
        # sim_train = sem_meta.drop(sim_val.index)
        
        dataset_train = Dataset_SIM(data_dir=sem_meta,
                                transform=transform_train, task=task,
                                data_type='both')

        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        num_data_train = len(dataset_train)
        num_batch_train = np.ceil(num_data_train / batch_size)

    ## Network setting

    if network == 'cyclegan' :
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device) # UNET을 쓰라고 하는데?
        netG_b2a = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device) # PatchGAN으로 해야 함
        netD_b = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)
        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

    elif network == "DCGAN" :
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker,  norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    elif network == "pix2pix" : 
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)

    ## setup loss & optimization
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999)) # itertools : 두 개의 파라미터를 동시에 옵티마이즈
    optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999)) 

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = 'gray'

    ## setup tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))


    ## load from checkpoints
    st_epoch = 0

    if mode == 'train' or mode == "train_SIM" or mode == "train_sem_depth":
        if train_continue == 'on' :
            netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
                                                                                netG_a2b=netG_a2b, netG_b2a=netG_b2a,
                                                                                netD_a=netD_a, netD_b=netD_b,
                                                                                optimG=optimG, optimD=optimD)
        
        for epoch in range(st_epoch + 1, num_epoch + 1) :
            ## training phase
            netG_a2b.train()
            netG_b2a.train()
            netD_a.train()
            netD_b.train()

            loss_G_a2b_train = []       # adversal loss
            loss_G_b2a_train = []
            loss_D_a_train = []
            loss_D_b_train = []
            loss_cycle_a_train = []     # cycle loss
            loss_cycle_b_train = []
            loss_ident_a_train = []     # identity loss
            loss_ident_b_train = []

            for batch, data in enumerate(loader_train, 1):  # start = 1
                # Forward pass
                # import ipdb; ipdb.set_trace()
                input_a = data['data_a'].to(device)    # REAL
                input_b = data['data_b'].to(device)    # SIM

                # forward netG
                output_b = netG_a2b(input_a)
                # from torchsummary import summary
                # summary(netG_a2b, input_size=(1, 72, 48))
                # import ipdb; ipdb.set_trace()
                recon_a = netG_b2a(output_b)

                output_a = netG_b2a(input_b)
                recon_b = netG_a2b(output_a)

                # backward netD
                set_requires_grad([netD_a, netD_b], True)
                optimD.zero_grad()

                # backward netD_a
                pred_real_a = netD_a(input_a)
                pred_fake_a = netD_a(output_a.detach())   # detach로 해서 모델에서 떼어준다.

                loss_D_a_real = fn_gan(pred_real_a, torch.ones_like(pred_real_a))     # torch.ones_like : True
                loss_D_a_fake = fn_gan(pred_fake_a, torch.zeros_like(pred_fake_a))    # torch.zeros_like : False
                loss_D_a = 0.5 * (loss_D_a_real + loss_D_a_fake)

                # backward netD_b
                pred_real_b = netD_b(input_b)
                pred_fake_b = netD_b(output_b.detach())   # detach로 해서 모델에서 떼어준다.

                loss_D_b_real = fn_gan(pred_real_b, torch.ones_like(pred_real_b))     # torch.ones_like : True
                loss_D_b_fake = fn_gan(pred_fake_b, torch.zeros_like(pred_fake_b))    # torch.zeros_like : False
                loss_D_b = 0.5 * (loss_D_b_real + loss_D_b_fake)

                loss_D = loss_D_a + loss_D_b
                loss_D.backward()
                optimD.step()

                # backward netG
                set_requires_grad([netD_a, netD_b], False)
                optimG.zero_grad()

                pred_fake_a = netD_a(output_a)
                pred_fake_b = netD_b(output_b)

                loss_G_a2b = fn_gan(pred_fake_a, torch.ones_like(pred_fake_a))
                loss_G_b2a = fn_gan(pred_fake_b, torch.ones_like(pred_fake_b))

                loss_cycle_a = fn_cycle(input_a, recon_a)
                loss_cycle_b = fn_cycle(input_b, recon_b)

                ident_a = netG_b2a(input_a)
                ident_b = netG_a2b(input_b)

                loss_ident_a = fn_ident(input_a, ident_a)
                loss_ident_b = fn_ident(input_b, ident_b)

                loss_G = (loss_G_a2b + loss_G_b2a) + \
                         wgt_cycle * (loss_cycle_a + loss_cycle_b) + \
                         wgt_cycle * wgt_ident * (loss_ident_a + loss_ident_b)

                loss_G.backward()
                optimG.step()

                # get losses
                loss_G_a2b_train += [loss_G_a2b.item()]
                loss_G_b2a_train += [loss_G_b2a.item()]

                loss_D_a_train += [loss_D_a.item()]
                loss_D_b_train += [loss_D_b.item()]

                loss_cycle_a_train += [loss_cycle_a.item()]
                loss_cycle_b_train += [loss_cycle_b.item()]

                loss_ident_a_train += [loss_ident_a.item()]
                loss_ident_b_train += [loss_ident_b.item()]

                print("TRAIN: EPOCH %03d / %03d | BATCH %04d / %04d | "
                      "GEN a2b %.4f b2a %.4f | "
                      "DISC a %.4f b %.4f | "
                      "CYCLE a %.4f b %.4f | "
                      "IDENT a %.4f b %.4f | " %
                      (epoch, num_epoch, batch, num_batch_train,
                       np.mean(loss_G_a2b_train), np.mean(loss_G_b2a_train),
                       np.mean(loss_D_a_train), np.mean(loss_D_b_train),
                       np.mean(loss_cycle_a_train), np.mean(loss_cycle_b_train),
                       np.mean(loss_ident_a_train), np.mean(loss_ident_b_train)))

                if batch % 300 == 0 :
                # if batch % 10 == 0 :
                    # Tensorboard
                    # import ipdb; ipdb.set_trace()
                    # input과 label 모두 이미지로 되어 있음. 이미지로 저장한다.
                    # input_a = fn_tonumpy(fn_denorm(input_a, mean=0.5, std=0.5)).squeeze()
                    # input_b = fn_tonumpy(fn_denorm(input_b, mean=0.5, std=0.5)).squeeze()
                    # output_a = fn_tonumpy(fn_denorm(output_a, mean=0.5, std=0.5)).squeeze()
                    # output_b = fn_tonumpy(fn_denorm(output_b, mean=0.5, std=0.5)).squeeze()

                    input_a = fn_tonumpy(input_a).squeeze()
                    input_b = fn_tonumpy(input_b).squeeze()
                    output_a = fn_tonumpy(output_a).squeeze()
                    output_b = fn_tonumpy(output_b).squeeze()

                    input_a = np.clip(input_a, a_min=0, a_max=1)
                    input_b = np.clip(input_b, a_min=0, a_max=1)
                    output_a = np.clip(output_a, a_min=0, a_max=1)
                    output_b = np.clip(output_b, a_min=0, a_max=1)

                    input_a = Image.fromarray(input_a[0]*255.0)
                    input_b = Image.fromarray(input_b[0]*255.0)
                    output_a = Image.fromarray(output_a[0]*255.0)
                    output_b = Image.fromarray(output_b[0]*255.0)

                    input_a = input_a.convert('L')
                    input_b = input_b.convert('L')
                    output_a = output_a.convert('L')
                    output_b = output_b.convert('L')

                    id = num_batch_train * (epoch - 1) + batch

                    if mode == 'train' :
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_input_REAL.png' % (epoch, id)), input_a[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_input_SIM.png' % (epoch, id)), input_b[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_output_REAL.png' % (epoch, id)), output_a[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_output_SIM.png' % (epoch, id)), output_b[0], cmap=cmap)
                        input_a.save(os.path.join(result_dir_train, 'png', '%03d_%04d_input_REAL.png' % (epoch, id)), cmap=cmap)
                        input_b.save(os.path.join(result_dir_train, 'png', '%03d_%04d_input_SIM.png' % (epoch, id)), cmap=cmap)
                        output_a.save(os.path.join(result_dir_train, 'png', '%03d_%04d_output_REAL.png' % (epoch, id)), cmap=cmap)
                        output_b.save(os.path.join(result_dir_train, 'png', '%03d_%04d_output_SIM.png' % (epoch, id)), cmap=cmap)

                    elif mode == "train_SIM" or mode == "train_sem_depth" :
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_input_SEM.png' % (epoch, id)), input_a[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_input_Depth.png' % (epoch, id)), input_b[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_output_SEM.png' % (epoch, id)), output_a[0], cmap=cmap)
                        # plt.imsave(os.path.join(result_dir_train, 'png', '%03d_%04d_output_Depth.png' % (epoch, id)), output_b[0], cmap=cmap)
                        input_a.save(os.path.join(result_dir_train, 'png', '%03d_%04d_input_SEM.png' % (epoch, id)), cmap=cmap)
                        input_b.save(os.path.join(result_dir_train, 'png', '%03d_%04d_input_Depth.png' % (epoch, id)), cmap=cmap)
                        output_a.save(os.path.join(result_dir_train, 'png', '%03d_%04d_output_SEM.png' % (epoch, id)), cmap=cmap)
                        output_b.save(os.path.join(result_dir_train, 'png', '%03d_%04d_output_Depth.png' % (epoch, id)), cmap=cmap)

                    # input_a = input_a.reshape(input_a.shape[0], input_a.shape[1], input_a.shape[2], 1)
                    # input_b = input_b.reshape(input_b.shape[0], input_b.shape[1], input_b.shape[2], 1)
                    # output_a = output_a.reshape(output_a.shape[0], output_a.shape[1], output_a.shape[2], 1)
                    # output_b = output_b.reshape(output_b.shape[0], output_b.shape[1], output_b.shape[2], 1)

                    # writer_train.add_image('input_a', input_a, id, dataformats='NHWC')
                    # writer_train.add_image('input_b', input_b, id, dataformats='NHWC')
                    # writer_train.add_image('output_a', output_a, id, dataformats='NHWC')
                    # writer_train.add_image('output_b', output_b, id, dataformats='NHWC')
        
            writer_train.add_scalar('loss_G_a2b', np.mean(loss_G_a2b_train), epoch)
            writer_train.add_scalar('loss_G_b2a', np.mean(loss_G_b2a_train), epoch)
            writer_train.add_scalar('loss_D_a', np.mean(loss_D_a_train), epoch)
            writer_train.add_scalar('loss_D_b', np.mean(loss_D_b_train), epoch)
            writer_train.add_scalar('loss_cycle_a', np.mean(loss_cycle_a_train), epoch)
            writer_train.add_scalar('loss_cycle_b', np.mean(loss_cycle_b_train), epoch)
            writer_train.add_scalar('loss_ident_a', np.mean(loss_ident_a_train), epoch)
            writer_train.add_scalar('loss_ident_b', np.mean(loss_ident_b_train), epoch)

            # validation 부분 없음 !!

            if epoch % 1 == 0 or epoch == num_epoch:
                save(ckpt_dir=ckpt_dir, netG_a2b=netG_a2b, netG_b2a=netG_b2a, netD_a=netD_a, netD_b=netD_b, optimG=optimG, optimD=optimD, epoch=epoch)

        writer_train.close()

def test(args) :
    mode = args.mode
    train_continue = args.train_continue

    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    task = args.task
    opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker

    wgt_cycle = args.wgt_cycle
    wgt_ident = args.wgt_ident
    norm = args.norm

    network = args.network
    learning_type = args.learning_type

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("task: %s" % task)
    print("opts: %s" % opts)

    print("network: %s" % network)
    print("learning type: %s" % learning_type)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## setup dataset
    result_dir_test = os.path.join(result_dir, 'test')
    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'png'))
    
    ## Test Network
    if mode == "test" :
        # transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)]) # Jitter 안 함
        transform_test = transforms.Compose([])

        sem_test_meta = pd.read_csv(data_dir+"\\test_sem_meta.csv")

        dataset_test_a = Dataset(data_dir=sem_test_meta, transform=transform_test, data_type='a')
        loader_test_a = DataLoader(dataset_test_a, batch_size=batch_size, shuffle=False, num_workers=8)

        num_data_test_a = len(dataset_test_a)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

        # dataset_test_b = Dataset(data_dir=sem_test_meta, transform=transform_test, data_type='b')
        # loader_test_b = DataLoader(dataset_test_b, batch_size=batch_size, shuffle=False, num_workers=8)

        # num_data_test_b = len(dataset_test_b)
        # num_batch_test_b = np.ceil(num_data_test_b / batch_size)

    elif mode == "test_SIM":
        transform_test = transforms.Compose([])

        sim_test_meta = pd.read_csv(data_dir+"\\test_sim_sem_meta.csv")

        dataset_test_a = Dataset_SIM(data_dir=sim_test_meta, transform=transform_test, data_type='a')
        loader_test_a = DataLoader(dataset_test_a, batch_size=batch_size, shuffle=False, num_workers=8)

        num_data_test_a = len(dataset_test_a)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

    elif  mode == "test_sem_depth"  :
        transform_test = transforms.Compose([ ])    
        
        sim_test_meta = pd.read_csv(data_dir+"\\test_sem_meta.csv") 
        
        dataset_train = Dataset(data_dir=sim_test_meta, transform=transform_test, data_type='a')

        loader_test_a = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=8)

        num_data_test_a = len(dataset_train)
        num_batch_test_a = np.ceil(num_data_test_a / batch_size)

    ## setup network
    if network == "cyclegan" :
        netG_a2b = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netG_b2a = CycleGAN(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)

        netD_a = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)
        netD_b = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG_a2b, init_type='normal', init_gain=0.02)
        init_weights(netG_b2a, init_type='normal', init_gain=0.02)
        init_weights(netD_a, init_type='normal', init_gain=0.02)
        init_weights(netD_b, init_type='normal', init_gain=0.02)

    elif network == "DCGAN" :
        netG = DCGAN(in_channels=100, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)
    
    elif network == "pix2pix" :
        netG = Pix2Pix(in_channels=nch, out_channels=nch, nker=nker, norm=norm).to(device)
        netD = Discriminator(in_channels=2 * nch, out_channels=1, nker=nker, norm=norm).to(device)

        init_weights(netG, init_type='normal', init_gain=0.02)
        init_weights(netD, init_type='normal', init_gain=0.02)


    # loss function
    fn_cycle = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    fn_ident = nn.L1Loss().to(device)

    # optimizer
    optimG = torch.optim.Adam(itertools.chain(netG_a2b.parameters(), netG_b2a.parameters()), lr=lr, betas=(0.5, 0.999))  # itertools : 두 개의 파라미터를 동시에 옵티마이즈
    optimD = torch.optim.Adam(itertools.chain(netD_a.parameters(), netD_b.parameters()), lr=lr, betas=(0.5, 0.999))  

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean

    cmap = 'gray'
    ## Train Network
    st_epoch = 0

    if mode == "test" or mode == "test_SIM" or mode == "test_sem_depth":
        netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, 
                                                                            netG_a2b=netG_a2b, netG_b2a=netG_b2a, 
                                                                            netD_a=netD_a, netD_b=netD_b,
                                                                            optimG=optimG, optimD=optimD)

        with torch.no_grad() :
            netG_a2b.eval()
            netG_b2a.eval()

            for batch, data in enumerate(loader_test_a, 1) :
                # forward pass
                input_a = data['data_a'].to(device)
                output_b = netG_a2b(input_a)

                # Tensorboard
                # input_a = fn_tonumpy(fn_denorm(input_a, mean=0.5, std=0.5)).squeeze()
                # output_b = fn_tonumpy(fn_denorm(output_b, mean=0.5, std=0.5)).squeeze()
                # import ipdb; ipdb.set_trace()

                input_a = fn_tonumpy(input_a).squeeze()
                output_b = fn_tonumpy(output_b).squeeze()


                for j in range(input_a.shape[0]) :
                    id = batch_size * (batch - 1) + j

                    input_a_ = input_a[j]
                    output_b_ = output_b[j]

                    input_a_ = np.clip(input_a_, a_min=0, a_max=1)
                    output_b_ = np.clip(output_b_, a_min=0, a_max=1)

                    input_a_ = Image.fromarray(input_a_*255.0)
                    output_b_ = Image.fromarray(output_b_*255.0)

                    input_a_ = input_a_.convert('L')
                    output_b_ = output_b_.convert('L')

                    # plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_a.png' % id), input_a_, cmap=cmap)
                    # plt.imsave(os.path.join(result_dir_test, 'png', '%05d_output_b.png' % id), output_b_, cmap=cmap)
                    # plt.imsave(os.path.join(result_dir_test, 'png', '%06d.png' % id), output_b_, cmap=cmap)
                    output_b_.save(os.path.join(result_dir_test, 'png', '%06d.png' % id), cmap=cmap)


                print("TEST A: BATCH %04d / %04d | " % (id + 1, num_data_test_a))
            

            # for batch, data in enumerate(loader_test_b, 1) :
            #     # forward pass
            #     input_b = data['data_b'].to(device)
            #     output_a = netG_b2a(input_b)

            #     # Tensorboard
            #     input_b = fn_tonumpy(fn_denorm(input_b, mean=0.5, std=0.5)).squeeze()
            #     output_a = fn_tonumpy(fn_denorm(output_a, mean=0.5, std=0.5)).squeeze()

            #     for j in range(input_b.shape[0]) : 
            #         id = batch_size * (batch - 1) + j

            #         input_b_ = input_b[j]
            #         output_a_ = output_a[j]

            #         input_b_ = np.clip(input_b_, a_min=0, a_max=1)
            #         output_a_ = np.clip(output_a_, a_min=0, a_max=1)

            #         plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input_b.png' % id), input_b_, cmap=cmap)
            #         plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output_a.png' % id), output_a_, cmap=cmap)

                # print("TEST B: BATCH %04d / %04d | " % (id + 1, num_data_test_b))

            