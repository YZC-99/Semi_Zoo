import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset, IDRIDDataset
import torch
import glob
import argparse
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
from utils.losses import OhemCrossEntropy, annealing_softmax_focalloss, softmax_focalloss, weight_softmax_focalloss
from utils.test_utils import DR_metrics, Sklearn_DR_metrics
from utils.util import color_map, gray_to_color
import random
from utils.util import get_optimizer, PolyLRwithWarmup, compute_sdf, compute_sdf_luoxd, compute_sdf_multi_class
from utils.scheduler.my_scheduler import my_decay_v1
from utils.bulid_model import build_model
from PIL import Image
from utils.training_utils import criteria
from dataloader.transform import cutmix
import time
import logging
import numpy as np
import os
import shutil
import logging
import math
from torchvision import transforms
import torchvision.transforms.functional as F
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from utils.scheduler.poly_lr import poly_epoch_decay, poly_epoch_decay_v2, poly_epoch_decay_v3
import ttach as tta

parser = argparse.ArgumentParser()

# ==============model===================
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--backbone', type=str, default='b2')
parser.add_argument('--encoder_deepth', type=int, default=5)
parser.add_argument('--fpn_out_c', type=int, default=48, help='the out-channels of the FPN module')
parser.add_argument('--fpn_pretrained', action='store_true')
parser.add_argument('--sr_out_c', type=int, default=128, help='the out-channels of the SR module')
parser.add_argument('--sr_pretrained', action='store_true')
parser.add_argument('--decoder_attention_type', type=str, default=None,
                    choices=['scse', 'sccbam', 'sc2cbam', 'scbam', 'sc2bam'])
parser.add_argument('--ckpt_weight', type=str, default=None)
parser.add_argument('--exclude_keys', type=str, default=None)
# ==============model===================

# ==============loss===================
parser.add_argument('--main_criteria', type=str, default='ce',
                    choices=['ce', 'weight-ce', 'dice', 'ce-dice', 'blv', 'softmax_focal_blv', 'softmax_focal',
                             'annealing_softmax_focal'])
parser.add_argument('--obj_loss', type=float, default=-1.0)
parser.add_argument('--ce_weight', type=float, nargs='+', default=[1, 1, 1, 1, 1], help='List of floating-point values')
parser.add_argument('--cls_num_list', type=float, nargs='+', default=[645688315, 704963, 6498614, 5319349, 1248855],
                    help='')
parser.add_argument('--ohem', type=float, default=-1.0)
parser.add_argument('--annealing_softmax_focalloss', action='store_true')
parser.add_argument('--softmax_focalloss', action='store_true')
parser.add_argument('--weight_softmax_focalloss', action='store_true')
parser.add_argument('--with_dice', action='store_true')
# ==============loss===================

# ==============lr===================
parser.add_argument('--base_lr', type=float, default=0.00025)
parser.add_argument('--lr_decouple', action='store_true')
parser.add_argument('--warmup', type=float, default=0.01)
parser.add_argument('--scheduler', type=str, default='poly-v1',
                    choices=['poly-v1', 'poly-v2', 'poly-v3', 'poly', 'no', 'my_decay_v1'])
# ==============lr===================

# ==============training params===================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--exp', type=str, default='IDRID')
parser.add_argument('--save_period', type=int, default=5000)
parser.add_argument('--val_period', type=int, default=100)
parser.add_argument('--dataset_name', type=str, default='IDRID')
parser.add_argument('--CLAHE', type=int, default=2)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--max_iterations', type=int, default=10000)
parser.add_argument('--autodl', action='store_true')
parser.add_argument('--cutmix_prob', default=-1.0)






args = parser.parse_args()
snapshot_path = "./exp_refer_2d_dr/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr

if __name__ == '__main__':

    model = 'Unet_wTri'
    # backbone = 'se_resnet50'
    backbone = 'mobileone_s0'
    # backbone = 'vgg11'
    # init model
    model = build_model(args, model=model, backbone=backbone, in_chns=3, class_num1=args.num_classes,
                        class_num2=2, fuse_type=None, ckpt_weight=args.ckpt_weight)

    # 计算模型大小
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = model_parameters * 4  # float32参数每个占用4字节
    model_size_mb = model_size / (1024 * 1024)  # 转换为MB

    print(f'模型大小: {model_size_mb:.2f} MB')



