import os
import numpy as np
from scipy.stats import poisson
from skimage.transform import rescale, resize

import torch
import torch.nn as nn

class Parser : 
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

        # set gpu ids
        str_ids = self.__args.gpu_ids.split(',')
        self.__args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.__args.gpu_ids.append(id)
        # if len(self.__args.gpu_ids) > 0:
        #     torch.cuda.set_device(self.__args.gpu_ids[0])
    
    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args
    
    def write_args(self):
        params_dict = vars(self.__args) # vars :  __dict__ attribute를 돌려준다.

        log_dir = os.path.join(params_dict['dir_log'], params_dict['scope'], params_dict['name_data'])
        args_name = os.path.join(log_dir, 'args.txt')


        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        with open(args_name, 'wt') as args_fid:
            args_fid.write('----' * 10 + '\n')
            args_fid.write('{0:^40}'.format('PARAMETER TABLES') + '\n')
            args_fid.write('----' * 10 + '\n')
            for k, v in sorted(params_dict.items()):
                args_fid.write('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)) + '\n')
            args_fid.write('----' * 10 + '\n')

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)

        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)

# class Logger :
#     def __init__(self, info=logging.INFO, name=__name__):
#         logger = logging.getLogger(name)    # getLogger : 해당 이름에 해당하는 logger를 전달받는다.
#         logger.setLevel(info)   # setLevel : 로그 레벨을 설정한다. (INFO : 작업이 정상적으로 작동하고 있다는 확인 메시지)

#         self.__logger = logger

#     def get_logger(self, handler_type='stream_handler') :
#         if handler_type == 'stream_handler' :
#             handler = logging.StreamHandler()   # StreamHandler : 핸들러란 내가 로깅한 정보가 출력되는 위치 설정하는 것
#             log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Formatter : format에 추가한 정보를 모든 로그 데이터에 추가해서 출력
#             handler.setFormatter(log_format)
#         else :
#             handler = logging.FileHandler('utils.log')  # FileHandler : 파일을 만들어서 로그를 기록한다.
        
#         self.__logger.addHandler(handler)

#         return self.__logger

# 네트워크 grad 설정
def set_requires_grad(nets, requires_grad=False) :
    """Set requies_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list) :
        nets = [nets]
    for net in nets :
        if net is not None :
            for param in net.parameters() :
                param.requrires_grad = requires_grad

# weight 초기화
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
        classname = m.__class__.__name__    # 클래스 이름을 참조한다.
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # Conv2d, ConvTranspose2d, Linear 초기화
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

# Save Network
def save(ckpt_dir, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD, epoch) :
    if not os.path.exists(ckpt_dir) :
        os.makedirs(ckpt_dir)
    
    torch.save({'netG_a2b': netG_a2b.state_dict(), 'netG_b2a': netG_b2a.state_dict(), 
                'netD_a': netD_a.state_dict(), 'netD_b': netD_b.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# Load Network
def load(ckpt_dir, netG_a2b, netG_b2a, netD_a, netD_b, optimG, optimD) :
    if not os.path.exists(ckpt_dir):
        epoch = 0 
        return netG_a2b, netG_b2a, netD_a, netD_b, optimG,optimD, epoch
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))   # isdigit : True if all the characters are digits, otherwise False.
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_a2b.load_state_dict(dict_model['netG_a2b'])
    netG_b2a.load_state_dict(dict_model['netG_b2a'])
    netD_a.load_state_dict(dict_model['netD_a'])
    netD_b.load_state_dict(dict_model['netD_b'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_a2b, netG_b2a, netD_a, netD_b, optimG,optimD, epoch
    
# Add Sampling : "inpainting" 아트 워크의 손상, 품질 저하 또는 누락 된 부분을 채워 완전한 이미지를 제공하는 보존 프로세스
def add_sampling(img, type="random", opts=None):
    sz = img.shape

    if type == 'uniform' :
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(img.shape)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk
    
    elif type == "gaussian":
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        # gaus = a * np.exp(-((x - x0) ** 2 / (2 * sgmx ** 2) + (y - y0) ** 2 / (2 * sgmy ** 2)))
        # gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))
        # rnd = np.random.rand(sz[0], sz[1], 1)
        # msk = (rnd < gaus).astype(np.float)
        # msk = np.tile(msk, (1, 1, sz[2]))

        dst = img * msk

    return dst

    
## Add Noise
def add_noise(img, type="random", opts=None):
    sz = img.shape

    if type == "random" :
        sgm = opts[0]
        noise = sgm / 255.0 * np.random.randn(sz[0], sz[1], sz[2])
        dst = img + noise
    
    elif type == "poisson" :
        dst = poisson.rvs(255.0 * img) / 255.0
        noise = dst - img
    
    return dst

## Add blurring
def add_blur(img, type="bilinear", opts=None):
    if type == "nearest" :
        order = 0
    elif type == "bilinear" :
        order = 1
    elif type == "biquadratic" :
        order = 2
    elif type == "bicubic":
        order = 3
    elif type == "biquartic":
        order = 4
    elif type == "biquintic":
        order = 5
    
    sz = img.shape
    if len(opts) == 1 :
        keepdim = True
    else : 
        keepdim = opts[1]
    
    # dw = 1.0 / opts[0]
    # dst = rescale(img, scale=(dw, dw, 1), order=order)
    dst = resize(img, output_shape=(sz[0] // opts[0], sz[1] // opts[0], sz[2]), order=order)

    if keepdim:
        # dst = rescale(dst, scale=(1 / dw, 1 / dw, 1), order=order)
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst
