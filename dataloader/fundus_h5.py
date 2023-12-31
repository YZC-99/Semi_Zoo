import torch
# from dataloader.transform import  hflip,vflip, normalize, resize, random_scale_and_crop
# from dataloader.transform import random_rotate,random_translate
from dataloader.transforms_multi import resize,random_flip,random_rotate,random_scale_crop,normalize
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

def CLAHE(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用CLAHE
    clahe_result = clahe.apply(image)
    # 将OpenCV数组转换为Pillow图像
    clahe_rgb = Image.merge('RGB', [Image.fromarray(clahe_result)] * 3)
    return  clahe_rgb


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



    def get_item_h5(self,item):
        id = self.ids[item]
        name = id.split(' ')[0].split('.')[0].split('/')[-1]
        h5_path = os.path.join(self.root, (name + ".h5"))
        h5f = h5py.File(h5_path,'r')
        img = np.array(h5f['data']['image'])
        mask = np.array(h5f['data']['label'])
        contour = np.array(h5f['data']['contours']).sum(axis=0)
        h5f.close()

        sample = {'image':img,'label':mask,'contour':contour}


        if self.mode == 'semi_train':
            if random.random() > 0.5:
                sample =  random_flip(sample)
            if random.random() > 0.5:
                sample = random_rotate(sample)
            if random.random() > 0.5:
                sample = random_scale_crop(sample)

        # img, mask = resize(img, mask, self.size)
        sample = resize(sample, self.size)
        sample = normalize(sample)

        return sample

    def __getitem__(self, item):

        sample = self.get_item_h5(item)
        return sample

    def __len__(self):
        return len(self.ids)


# class Org_SemiDataset(Dataset):
#     def __init__(self,name, root, mode, size,
#                  id_path=None,h5_file=False,CLAHE = False,preprocess = False):
#         """
#         :param name: dataset name, pascal or cityscapes
#         :param root: root path of the dataset.
#         :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
#                      label: pseudo labeling the remaining unlabeled images.
#                      semi_train: semi-supervised learning with both labeled and unlabeled images.
#                      val: validation.
#
#         :param size: crop size of training images.
#         :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
#         :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
#         :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
#         """
#
#         self.name = name
#         self.root = root
#         self.mode = mode
#         self.size = size
#         self.h5_file = h5_file
#         self.CLAHE = CLAHE
#         self.preprocess = preprocess
#
#         if mode == 'semi_train':
#             id_path = '%s/%s' %(name,id_path)
#         elif mode == 'val':
#             id_path = '%s/val.txt' % name
#         elif mode == 'test':
#             id_path = '%s/test.txt' % name
#
#         with open(id_path, 'r') as f:
#             self.ids = f.read().splitlines()
#
#     def get_item_nor(self,item):
#         id = self.ids[item]
#         img_path = os.path.join(self.root, id.split(' ')[0])
#         if self.CLAHE:
#             img = CLAHE(img_path)
#         else:
#             img = Image.open(img_path)
#
#         if "DDR" in id or "G1020" in id or "ACRIMA" in id:
#             mask = Image.fromarray(np.zeros((2, 2)))
#         elif "HRF" in id:
#             mask_path = os.path.join(self.root, id.split(' ')[1])
#             mask = Image.open(mask_path).convert('L')
#             mask_arr = np.array(mask) / 255
#             mask_arr[mask_arr > 2] = 0
#             mask = Image.fromarray(mask_arr)
#             # print(np.unique(np.array(mask)))
#         else:
#             mask_path = os.path.join(self.root, id.split(' ')[1])
#             mask = Image.open(mask_path)
#
#         if self.mode == 'semi_train':
#
#
#             img, mask = hflip(img, mask, p=0.5)
#             img, mask = vflip(img, mask, p=0.5)
#             # img, mask = random_rotate(img, mask, p=0.5)
#             img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8,
#                                               max_scale=1.2, p=0.0)
#
#         img, mask = resize(img, mask, self.size)
#         img, mask = normalize(img, mask)
#         if self.preprocess:
#             image_edges_info = np.load(img_path.replace('images_cropped','img2canny-dog2npy').replace('jpg','npy'),allow_pickle=True)
#             image_edges_info = image_edges_info / 255
#             image_edges_info = torch.from_numpy(image_edges_info)
#
#
#         if mask is not None:
#             mask[mask > 2] = 0
#         return {'image': img, 'label': mask}
#
#     def get_item_h5(self,item):
#         id = self.ids[item]
#         name = id.split(' ')[0].split('.')[0]
#         h5_path = os.path.join(self.root, (name + ".h5"))
#         h5f = h5py.File(h5_path,'r')
#         img = h5f['image']
#         if "DDR" in id or "G1020" in id or "ACRIMA" in id:
#             mask = Image.fromarray(np.zeros((2, 2)))
#             contour = Image.fromarray(np.zeros((2, 2)))
#         else:
#             mask = h5f['label']
#             contour = h5f['contour']
#
#         if self.mode == 'semi_train':
#             if random.random() > 0.5:
#                 img,mask,contour =  random_rot_flip(img,mask,contour)
#             elif random.random() > 0.5:
#                 img, mask, contour = random_rotate(img, mask, contour)
#
#         # img, mask = resize(img, mask, self.size)
#         img, mask, contour = resize_numpy(img, mask, contour, self.size)
#         img, mask = normalize(img, mask)
#
#         if mask is not None:
#             mask[mask > 2] = 0
#         return {'image': img, 'label': mask}
#
#     def __getitem__(self, item):
#         if not self.h5_file:
#             sample = self.get_item_nor(item)
#         else:
#             sample = self.get_item_h5(item)
#         return sample
#
#     def __len__(self):
#         return len(self.ids)





