import torch
from dataloader.transform import  hflip,vflip, normalize, resize, random_scale_and_crop,resize1440
from dataloader.transform import random_rotate,random_translate,random_scale
from dataloader.transforms_np import resize_np,random_flip_np,random_rotate_np,normalize_np
import cv2
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np
from scipy import ndimage
from torchvision.transforms import functional
import scipy.io






class SemiDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,h5_file=False,CLAHE = False,preprocess = False):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """

        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.h5_file = h5_file
        self.CLAHE = CLAHE
        self.preprocess = preprocess

        if mode == 'semi_train':
            id_path = '%s/%s' %(name,id_path)
        elif mode == 'val':
            id_path = '%s/val.txt' % name
        elif mode == 'test':
            id_path = '%s/test.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_item_nor(self,item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        # if self.CLAHE:
        #     img = CLAHE(img_path)
        # else:
        img = Image.open(img_path)

        if "DDR" in id or "G1020" in id or "ACRIMA" in id:
            mask = Image.fromarray(np.zeros((2, 2)))
        elif "HRF" in id:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask) / 255
            mask_arr[mask_arr > 2] = 0
            mask = Image.fromarray(mask_arr)
            # print(np.unique(np.array(mask)))
        else:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path)

        if self.mode == 'semi_train':


            img, mask = hflip(img, mask, p=0.5)
            img, mask = vflip(img, mask, p=0.5)
            img, mask = random_rotate(img, mask, p=0.5)
            img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8,
                                              max_scale=1.2, p=0.0)

        img, mask = resize(img, mask, self.size)
        img, mask = normalize(img, mask)
        if self.preprocess:
            image_edges_info = np.load(img_path.replace('images_cropped','img2canny-dog2npy').replace('jpg','npy'),allow_pickle=True)
            image_edges_info = image_edges_info / 255
            image_edges_info = torch.from_numpy(image_edges_info)
        return {'image': img, 'label': mask}


    def __getitem__(self, item):
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)

class IDRIDDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,CLAHE = False):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.CLAHE = CLAHE

        if mode == 'semi_train':
            id_path = '%s/%s' %(name,id_path)
        elif mode == 'val':
            id_path = '%s/val.txt' % name
        elif mode == 'test':
            id_path = '%s/test.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_item_nor(self,item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        if self.CLAHE:
            img_path = img_path.replace('/images','/images_clahe')
            img = Image.open(img_path)
        else:
            img = Image.open(img_path)

        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path)

        if self.mode == 'semi_train':
            img, mask = hflip(img, mask, p=0.5)
            img, mask = vflip(img, mask, p=0.5)
            img, mask = random_rotate(img, mask,p=0.5,max_rotation_angle=30)
            img, mask = random_scale(img,mask,p=0.5)

        if self.size == 1440:
            img, mask = resize1440(img, mask)
        else:
            img, mask = resize(img, mask, self.size)
        if self.CLAHE:
            img, mask = normalize(img, mask, mean=(0.0,), std=(1.0,))
        else:
            img, mask = normalize(img, mask,mean=(116.51,56,437,16.31),std=(81.60,41.72,7.36))

        return {'image': img, 'label': mask}

    def __getitem__(self, item):
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)


class DDRDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,CLAHE = False):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.CLAHE = CLAHE

        if mode == 'semi_train':
            id_path = '%s/%s' %(name,id_path)
        elif mode == 'val':
            id_path = '%s/val.txt' % name
        elif mode == 'test':
            id_path = '%s/test.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_item_nor(self,item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        if self.CLAHE:
            img_path = img_path.replace('/images','/images_clahe')
            img = Image.open(img_path)
        else:
            img = Image.open(img_path)

        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path)

        if self.mode == 'semi_train':
            img, mask = hflip(img, mask, p=0.5)
            img, mask = vflip(img, mask, p=0.5)
            img, mask = random_rotate(img, mask,p=0.5,max_rotation_angle=30)


        img, mask = resize(img, mask, self.size)
        img, mask = normalize(img, mask,mean=(91.78,56.94,25.83),std=(124.97,83.84,36.79))

        return {'image': img, 'label': mask}

    def __getitem__(self, item):
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)



