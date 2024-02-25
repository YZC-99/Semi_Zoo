import random
from tqdm import tqdm
from dataloader.fundus import SemiDataset,IDRIDDataset
import torch
import glob
import argparse
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss,MSELoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
import segmentation_models_pytorch as smp
from utils.losses import OhemCrossEntropy,annealing_softmax_focalloss,softmax_focalloss,weight_softmax_focalloss
from utils.test_utils import DR_metrics,Sklearn_DR_metrics
from utils.util import color_map,gray_to_color
import random
from utils.util import get_optimizer,PolyLRwithWarmup, compute_sdf,compute_sdf_luoxd,compute_sdf_multi_class
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
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from utils.scheduler.poly_lr import poly_epoch_decay,poly_epoch_decay_v2,poly_epoch_decay_v3
import ttach as tta

parser = argparse.ArgumentParser()

# ==============model===================
parser.add_argument('--model',type=str,default='unet')
parser.add_argument('--backbone',type=str,default='b2')
parser.add_argument('--encoder_deepth',type=int,default=5)
parser.add_argument('--fpn_out_c',type=int,default=-1,help='the out-channels of the FPN module')
parser.add_argument('--fpn_pretrained',action='store_true')
parser.add_argument('--sr_out_c',type=int,default=128,help='the out-channels of the SR module')
parser.add_argument('--sr_pretrained',action='store_true')
parser.add_argument('--decoder_attention_type',type=str,default=None,choices=['scse','sccbam','sc2cbam','scbam','sc2bam'])
parser.add_argument('--ckpt_weight',type=str,default=None)
parser.add_argument('--exclude_keys',type=str,default=None)
# ==============model===================

# ==============loss===================
parser.add_argument('--main_criteria', type=str, default='ce',choices=['ce','weight-ce','dice','ce-dice','blv','softmax_focal_blv','softmax_focal','annealing_softmax_focal'])
parser.add_argument('--obj_loss', type=float, default=-1.0)
parser.add_argument('--ce_weight', type=float, nargs='+', default=[1,1,1,1,1], help='List of floating-point values')
parser.add_argument('--cls_num_list', type=float, nargs='+', default=[645688315,704963,6498614,5319349,1248855], help='')
parser.add_argument('--ohem',type=float,default=-1.0)
parser.add_argument('--annealing_softmax_focalloss',action='store_true')
parser.add_argument('--softmax_focalloss',action='store_true')
parser.add_argument('--weight_softmax_focalloss',action='store_true')
parser.add_argument('--with_dice',action='store_true')
# ==============loss===================

# ==============lr===================
parser.add_argument('--base_lr',type=float,default=0.00025)
parser.add_argument('--lr_decouple',action='store_true')
parser.add_argument('--warmup',type=float,default=0.01)
parser.add_argument('--scheduler',type=str,default='poly-v1',choices=['poly-v1','poly-v2','poly-v3','poly','no','my_decay_v1'])
# ==============lr===================

# ==============training params===================
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--num_works',type=int,default=0)
parser.add_argument('--num_classes',type=int,default=5)
parser.add_argument('--exp',type=str,default='IDRID')
parser.add_argument('--save_period',type=int,default=5000)
parser.add_argument('--val_period',type=int,default=100)
parser.add_argument('--dataset_name',type=str,default='IDRID')
parser.add_argument('--CLAHE',type=int,default=2)
parser.add_argument('--optim',type=str,default='AdamW')
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--image_size',type=int,default=512)
parser.add_argument('--max_iterations',type=int,default=10000)
parser.add_argument('--autodl',action='store_true')
parser.add_argument('--cutmix_prob',default=-1.0)

# ==============test params===================
parser.add_argument('--tta',action='store_true')


def color_map_fn():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])
    cmap[2] = np.array([0, 255, 0])
    cmap[3] = np.array([0, 0, 255])
    cmap[4] = np.array([255, 255, 0])
    return cmap


tta_transforms = [transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.HorizontalFlip(),
                    transforms.ToTensor()]),
                  transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(90),
                    transforms.ToTensor()]),
                  transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(180),
                    transforms.ToTensor()]),
                  transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(270),
                    transforms.ToTensor()])
                 ]


def apply_tta_and_predict(model, image, tta_transforms, device):
    # 存储TTA预测结果
    tta_outputs = []

    # 原始图像预测
    original_pred = model(image.unsqueeze(0))
    tta_outputs.append(original_pred.cpu().data.numpy())

    # 应用TTA变换
    for tta_transform in tta_transforms:
        tta_image = tta_transform(image)
        tta_pred = model(tta_image.unsqueeze(0))
        tta_outputs.append(tta_pred.cpu().data.numpy())

    # 计算所有TTA预测结果的平均值
    mean_tta_output = np.mean(tta_outputs, axis=0)
    return mean_tta_output


def create_version_folder(snapshot_path):
    # 检查是否存在版本号文件夹
    version_folders = [name for name in os.listdir(snapshot_path) if name.startswith('version')]

    if not version_folders:
        # 如果不存在版本号文件夹，则创建version0
        new_folder = os.path.join(snapshot_path, 'version0')
    else:
        # 如果存在版本号文件夹，则创建下一个版本号文件夹
        last_version = max(version_folders)
        last_version_number = int(last_version.replace('version', ''))
        next_version = 'version{:02d}'.format(last_version_number + 1)
        new_folder = os.path.join(snapshot_path, next_version)

    os.makedirs(new_folder)
    return new_folder


args = parser.parse_args()
snapshot_path = "./exp_refer_2d_dr/" + args.exp + "/"
max_iterations = args.max_iterations
base_lr = args.base_lr


if __name__ == '__main__':

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    results_path = snapshot_path + "/" + 'results' + "/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_logs_path = snapshot_path + "/" + 'results-log.txt'

    device = "cuda:{}".format(args.device)

    # init model
    model = build_model(args,model=args.model,backbone=args.backbone,in_chns=3,class_num1=args.num_classes,class_num2=2,fuse_type=None,ckpt_weight=args.ckpt_weight)
    model.to(device)


    # init dataset
    root_base = '/home/gu721/yzc/data/dr/'
    if args.autodl:
        root_base = '/root/autodl-tmp/'


    # 验证集
    # init dataset
    val_dataset = IDRIDDataset(name='./dataset/{}'.format(args.dataset_name),
                                    # root="D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/REFUGE",
                                  root="{}{}".format(root_base,args.dataset_name),
                                  mode='val',
                                  size=args.image_size,
                                  CLAHE=args.CLAHE)

    val_loader = DataLoader(val_dataset,batch_size=1,num_workers=1)
    val_iteriter = tqdm(val_loader)

    model.eval()
    DR_val_metrics = DR_metrics(device)

    for i_batch,labeled_sampled_batch in enumerate(val_loader):
        time2 = time.time()
        labeled_batch, label_label_batch = labeled_sampled_batch['image'].to(device), labeled_sampled_batch['label'].to(device)
        image_id = labeled_sampled_batch['id'][0].split('/')[-1]

        if args.tta:
            outputs = apply_tta_and_predict(model, labeled_batch, tta_transforms, device)
        else:
            outputs = model(labeled_batch)
        out = torch.argmax(outputs,dim=1)
        out = out[0].detach().cpu().numpy().astype(np.int8)
        pred_mask = Image.fromarray(out,mode='P')
        pred_mask.putpalette(color_map_fn())
        pred_mask.save(results_path+'/'+image_id)

        DR_val_metrics.add(outputs.detach(),label_label_batch)
    with open(results_logs_path,'w') as f:
        val_metrics = DR_val_metrics.get_metrics()
        AUC_PR = val_metrics[0]
        AUC_ROC = val_metrics[1]
        Dice,IoU = val_metrics[2]['Dice'],val_metrics[2]['IoU']
        MA_AUC_PR, HE_AUC_PR, EX_AUC_PR, SE_AUC_PR = AUC_PR['MA_AUC_PR'],AUC_PR['HE_AUC_PR'],AUC_PR['EX_AUC_PR'],AUC_PR['SE_AUC_PR']
        MA_AUC_ROC, HE_AUC_ROC, EX_AUC_ROC, SE_AUC_ROC = AUC_ROC['MA_AUC_ROC'],AUC_ROC['HE_AUC_ROC'],AUC_ROC['EX_AUC_ROC'],AUC_ROC['SE_AUC_ROC']

        lines = [
            f'MA_AUC_PR:{MA_AUC_PR}\n',
            f'HE_AUC_PR:{HE_AUC_PR}\n',
            f'EX_AUC_PR:{EX_AUC_PR}\n',
            f'SE_AUC_PR:{SE_AUC_PR}\n',
            f'MA_AUC_ROC:{MA_AUC_ROC}\n',
            f'HE_AUC_ROC:{HE_AUC_ROC}\n',
            f'EX_AUC_ROC:{EX_AUC_ROC}\n',
            f'SE_AUC_ROC:{SE_AUC_ROC}\n',
            f'Dice:{Dice}\n',
            f'IoU:{IoU}\n'
        ]
        # 写入并打印每行
        for line in lines:
            f.write(line)
            print(line)  # 使用 strip() 移除尾随的换行符




