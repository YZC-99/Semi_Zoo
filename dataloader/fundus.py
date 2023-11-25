import torch
from dataloader.transform import crop,test_blur, hflip,vflip, normalize, resize, blur, cutout,dist_transform,random_scale_and_crop
from dataloader.transform import random_rotate,random_translate,add_salt_pepper_noise,random_scale,color_distortion
import cv2
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import scipy.io

class SemiDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None):
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

        if mode == 'semi_train':
            id_path = '%s/%s' %(name,id_path)
        elif mode == 'val':
            id_path = '%s/val.txt' % name
        elif mode == 'test':
            id_path = '%s/test.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        img = Image.open(img_path)
        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path)

        if self.mode == 'semi_train':
            img, mask = hflip(img, mask, p=0.5)
            img, mask = vflip(img, mask, p=0.5)
            # img, mask = random_rotate(img, mask, p=0.5)
            img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8, max_scale=1.2,p=0.0)

        img, mask = resize(img, mask, self.size)
        img, mask = normalize(img, mask)
        # boundary = dist_transform(mask)
        # return {'image':img, 'label':mask,'boundary':boundary}
        return {'image':img, 'label':mask}

    def __len__(self):
        return len(self.ids)
