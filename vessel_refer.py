from dataloader.fundus import SemiDataset, IDRIDDataset
import argparse
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.bulid_model import build_model
import time
import numpy as np
import os
import logging
import math
import torch
import sys
from utils.util import color_map,color_map_fn
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()

# ==============model===================
parser.add_argument('--model', type=str, default='UNet')
parser.add_argument('--backbone', type=str, default='resnet50')

parser.add_argument('--ckpt_weight', type=str,
                    default='/root/autodl-tmp/Semi_Zoo/exp_2d_vessel/DRIVE/UNet/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4/version01/iter_1000.pth')
parser.add_argument('--decoder_attention_type', type=str, default=None)
# ==============model===================


# ==============test params===================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--image_size', type=int, default=256)

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--exp', type=str, default='RIM-ONE')
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument('--dataset_name', type=str, default='RIM-ONE')

# parser.add_argument('--autodl',action='store_true')


parser.add_argument('--autodl', default=True, type=bool)


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
snapshot_path = "./exp_2d_refer_results/" + args.exp + "/"


def color_map_fn():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])
    cmap[2] = np.array([0, 255, 0])
    cmap[3] = np.array([0, 0, 255])
    cmap[4] = np.array([255, 255, 0])

    return cmap


if __name__ == '__main__':
    """
    1、搜寻snapshot_path下面的含有version的文件夹，如果没有就创建version0，即：
    snapshot_path = snapshot_path + "/" + "version0"
    2、如果有version的文件夹，如果当前文件夹下有version0和version01，则创建version02，以此类推
    """

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    snapshot_path = create_version_folder(snapshot_path)

    device = "cuda:{}".format(args.device)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # init model
    model = build_model(args, model=args.model, backbone=args.backbone, in_chns=3, class_num1=args.num_classes,
                        class_num2=2, fuse_type=None, ckpt_weight=args.ckpt_weight)
    model.to(device)

    # init dataset
    root_base = '/home/gu721/yzc/data/odoc/'
    if args.autodl:
        root_base = '/root/autodl-tmp/'

    labeled_dataset = SemiDataset(name='./dataset/{}'.format(args.dataset_name),
                                  root="{}{}".format(root_base, args.dataset_name),
                                  mode='semi_train',
                                  size=args.image_size,
                                  id_path='train.txt')

    labeledtrainloader = DataLoader(labeled_dataset,
                                    batch_size=1,
                                    num_workers=0,
                                    )

    model.eval()

    cmap = color_map_fn()
    for i_batch, labeled_sampled_batch in enumerate(labeledtrainloader):
        time2 = time.time()

        labeled_batch, label_label_batch = labeled_sampled_batch['image'].to(device), labeled_sampled_batch['label'].to(
            device)
        name = labeled_sampled_batch['name']
        all_batch = labeled_batch
        all_label_batch = label_label_batch

        outputs = model(all_batch)

        outputs_soft = torch.argmax(outputs, dim=1).squeeze().detach().cpu().numpy()
        print(np.unique(outputs_soft))

        name = name[0].split('/')[-1]
        outputs_soft = outputs_soft.astype(np.int8)
        pseudo_mask = Image.fromarray(outputs_soft, mode='P')
        pseudo_mask.putpalette(cmap)
        pseudo_mask.save(os.path.join(snapshot_path, name))

