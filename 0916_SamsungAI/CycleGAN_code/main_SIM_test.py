import argparse
import torch
import torch.backends.cudnn as cudnn

from train import *
from utils import *

# https://github.com/hanyoseob/youtube-cnn-007-pytorch-cyclegan

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # 라이브러리 중복 사용을 허용한다.

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test_SIM", choices=["train", "test","train_SIM", "test_SIM"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=2, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="D:\Data\\3D_Metrology\\SEM_cyclegan\\log_result_bc16\\test\\png", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="D:\\Data\\3D_Metrology\\SEM_cyclegan\\SIM_checkpoints_bc16", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="D:\\Data\\3D_Metrology\\SEM_cyclegan\\SIM_log_tensorboard_bc16", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="D:\\Data\\3D_Metrology\\SEM_cyclegan\\SIM_log_result_bc16", type=str, dest="result_dir")

parser.add_argument("--task", default="cyclegan", choices=['DCGAN', 'pix2pix', 'cyclegan'], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts') 

parser.add_argument("--ny", default=256, type=int, dest="ny")
parser.add_argument("--nx", default=256, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt_cycle", default=1e1, type=float, dest="wgt_cycle")   # cycle consistency loss
parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")   # identity mapping loss 
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--network", default="cyclegan", choices=['DCGAN', 'pix2pix', 'cyclegan'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train" or args.mode == "train_SIM":
        train(args)
    elif args.mode == "test" or args.mode == "test_SIM" :
        test(args)