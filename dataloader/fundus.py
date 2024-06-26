import copy

import torch
from dataloader.transform import  hflip,vflip, normalize, resize, random_scale_and_crop,resize1440
from dataloader.transform import random_rotate,random_translate,random_scale
from dataloader.transform_dr import *
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


MEAN_RGB = [0.485 , 0.456 , 0.406 ]
STDDEV_RGB = [0.229 , 0.224 , 0.225 ]



class SemiDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,
                 h5_file=False,
                 CLAHE = False,
                 preprocess = False,
                 pseudo_vessel = False ):
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
        self.pseudo_vessel = pseudo_vessel

        if mode == 'semi_train':
            id_path = '%s/%s' %(name,id_path)
        elif mode == 'val':
            id_path = '%s/val.txt' % name
        elif mode == 'test':
            id_path = '%s/test.txt' % name
        elif mode == 'all':
            id_path = '%s/all.txt' % name

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
        elif "HRF" in id or 'CHASEDB1' in id or 'DRIVE' in id:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask) / 255
            mask_arr[mask_arr > 2] = 0
            mask = Image.fromarray(mask_arr)
        else:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path)

        if self.mode == 'semi_train':

            if random.random() < 0.5:
                img, mask = hflip(img, mask)
            if random.random() < 0.5:
                img, mask = vflip(img, mask)
            if random.random() < 0.5:
                img, mask = random_rotate(img, mask)
            img, mask = resize(img, mask, self.size, self.size)
            if random.random() < 0.5:
                img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8,
                                              max_scale=1.2)
        else:
            img, mask = resize(img, mask, self.size,self.size)
        img, mask = normalize(img, mask, mean=MEAN_RGB, std=STDDEV_RGB)
        if self.preprocess:
            image_edges_info = np.load(img_path.replace('images_cropped','img2canny-dog2npy').replace('jpg','npy'),allow_pickle=True)
            image_edges_info = image_edges_info / 255
            image_edges_info = torch.from_numpy(image_edges_info)
        return {'image': img, 'label': mask,'name':id.split(' ')[1]}


    def __getitem__(self, item):
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)


class ODOC_Vessel_Dataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,
                 pseudo_vessel = False ):

        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_vessel = pseudo_vessel

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
        img = Image.open(img_path)

        if "HRF" in id or 'CHASEDB1' in id or 'DRIVE' in id:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask) / 255
            mask_arr[mask_arr > 2] = 0
            mask = Image.fromarray(mask_arr)
        else:
            mask_path = os.path.join(self.root, id.split(' ')[1])
            mask = Image.open(mask_path)

        if self.mode == 'semi_train':

            if random.random() < 0.5:
                img, mask = hflip(img, mask)
            if random.random() < 0.5:
                img, mask = vflip(img, mask)
            if random.random() < 0.5:
                img, mask = random_rotate(img, mask)
            img, mask = resize(img, mask, self.size, self.size)
            if random.random() < 0.5:
                img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8,
                                              max_scale=1.2)
        else:
            img, mask = resize(img, mask, self.size,self.size)
        img, mask = normalize(img, mask, mean=MEAN_RGB, std=STDDEV_RGB)

        return {'image': img, 'label': mask,'name':id.split(' ')[1]}

    def get_item_odoc_vessel(self, item):
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])
        img = Image.open(img_path)
        odoc_mask_path = os.path.join(self.root, id.split(' ')[1])
        odoc_mask = Image.open(odoc_mask_path)
        vessel_mask_path = odoc_mask_path.replace('my_gts_cropped','v_my_gts_cropped')
        vessel_mask = Image.open(vessel_mask_path)

        if random.random() < 0.5:
            img, odoc_mask, vessel_mask, _,_ = hflip_four(img, odoc_mask, vessel_mask)
        if random.random() < 0.5:
            img, odoc_mask, vessel_mask, _,_ = vflip_four(img, odoc_mask, vessel_mask)
        if random.random() < 0.5:
            img, odoc_mask, vessel_mask, _,_ = random_rotate_four(img, odoc_mask, vessel_mask)
        if random.random() < 0.5:
            img, odoc_mask, vessel_mask, _,_ = random_translate_four(img, odoc_mask, vessel_mask)
        img, odoc_mask, vessel_mask, _,_ = resize_four(img, odoc_mask, vessel_mask,height=self.size,width=self.size)
        if random.random() < 0.5:
            img, odoc_mask, vessel_mask, _,_ = random_scale_and_crop_four(img, odoc_mask, vessel_mask,
                                                                                 target_size=(self.size, self.size))
        img, odoc_mask, vessel_mask, _,_ = normalize_four(img, odoc_mask, vessel_mask, mean=MEAN_RGB, std=STDDEV_RGB)
        return {'image': img, 'odoc_label': odoc_mask,'vessel_mask':vessel_mask,'name':id.split(' ')[1]}

    def __getitem__(self, item):
        if self.pseudo_vessel:
            sample = self.get_item_odoc_vessel(item)
        else:
            sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)


class VesselDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,
                 h5_file=False,
                 CLAHE = False,
                 preprocess = False):
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


        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path).convert('L')
        mask_arr = np.array(mask) / 255
        mask_arr[mask_arr > 2] = 0
        mask = Image.fromarray(mask_arr)

        if self.mode == 'semi_train':

            if random.random() < 0.5:
                img, mask = hflip(img, mask)
            if random.random() < 0.5:
                img, mask = vflip(img, mask)
            if random.random() < 0.5:
                img, mask = random_rotate(img, mask)
            img, mask = resize(img, mask, self.size, self.size)
            if random.random() < 0.5:
                img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size), min_scale=0.8,
                                              max_scale=1.2)
        else:
            img, mask = resize(img, mask, self.size,self.size)
        img, mask = normalize(img, mask, mean=MEAN_RGB, std=STDDEV_RGB)
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


class BinaryIDRIDDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,CLAHE = 2):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = int(size)
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
        if self.CLAHE > 0:
            img_path = img_path.replace('/images','/images_clahe_{}'.format(self.CLAHE))
            img = Image.open(img_path)
        else:
            img = Image.open(img_path)
        org_img = copy.copy(img)
        fused_mask_path = os.path.join(self.root, id.split(' ')[1])
        MA_mask_path = fused_mask_path.replace("labels_fused","labels_indiv").\
                                        replace("/IDRiD","/1. Microaneurysms/IDRiD").\
                                        replace("_fuse.png","_MA.tif")
        HE_mask_path = fused_mask_path.replace("labels_fused","labels_indiv").\
                                        replace("/IDRiD","/2. Haemorrhages/IDRiD").\
                                        replace("_fuse.png","_HE.tif")
        EX_mask_path = fused_mask_path.replace("labels_fused","labels_indiv").\
                                        replace("/IDRiD","/3. Hard Exudates/IDRiD").\
                                        replace("_fuse.png","_EX.tif")
        SE_mask_path = fused_mask_path.replace("labels_fused","labels_indiv").\
                                        replace("/IDRiD","/4. Soft Exudates/IDRiD").\
                                        replace("_fuse.png","_SE.tif")
        # sample = {'image': img,
        #         'MA_mask': torch.zeros(1),
        #         'HE_mask': torch.zeros(1),
        #         'EX_mask': torch.zeros(1),
        #         'SE_mask': torch.zeros(1),
        #         }
        sample = {'image': img,
                'MA_mask': None,
                'HE_mask': None,
                'EX_mask': None,
                'SE_mask': None,
                }



        # if os.path.exists(MA_mask_path):
        #     MA_mask = Image.open(MA_mask_path)
        #     img, MA_mask = resize1440(org_img, MA_mask)
        #     img, MA_mask = normalize(img, MA_mask, mean=MEAN_RGB, std=STDDEV_RGB)
        #     sample['MA_mask'] = MA_mask
        # if os.path.exists(HE_mask_path):
        #     HE_mask = Image.open(HE_mask_path)
        #     img, HE_mask = resize1440(org_img, HE_mask)
        #     img, HE_mask = normalize(img, HE_mask, mean=MEAN_RGB, std=STDDEV_RGB)
        #     sample['HE_mask'] = HE_mask
        # if os.path.exists(EX_mask_path):
        #     EX_mask = Image.open(EX_mask_path)
        #     img, EX_mask = resize1440(org_img, EX_mask)
        #     img, EX_mask = normalize(img, EX_mask, mean=MEAN_RGB, std=STDDEV_RGB)
        #     sample['EX_mask'] = EX_mask
        # if os.path.exists(SE_mask_path):
        #     SE_mask = Image.open(SE_mask_path)
        #     img, SE_mask = resize1440(org_img, SE_mask)
        #     img, SE_mask = normalize(img, SE_mask, mean=MEAN_RGB, std=STDDEV_RGB)
        #     sample['SE_mask'] = SE_mask




        MA_mask = Image.open(MA_mask_path) if os.path.exists(MA_mask_path) else None

        HE_mask = Image.open(HE_mask_path) if os.path.exists(HE_mask_path) else None

        EX_mask = Image.open(EX_mask_path) if os.path.exists(EX_mask_path) else None

        SE_mask = Image.open(SE_mask_path) if os.path.exists(SE_mask_path) else None

        if self.mode == 'semi_train':
            if random.random() < 0.5:
                img,MA_mask,HE_mask,EX_mask,SE_mask = hflip_four(img,MA_mask,HE_mask,EX_mask,SE_mask)
            if random.random() < 0.5:
                img, MA_mask, HE_mask, EX_mask, SE_mask = vflip_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
            if random.random() < 0.5:
                img, MA_mask, HE_mask, EX_mask, SE_mask = random_rotate_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
            if random.random() < 0.5:
                img, MA_mask, HE_mask, EX_mask, SE_mask = random_translate_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
            if self.size == 1440:
                img, MA_mask, HE_mask, EX_mask, SE_mask = resize1440_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
            else:
                img, MA_mask, HE_mask, EX_mask, SE_mask = resize_four(img, MA_mask, HE_mask, EX_mask, SE_mask,width=self.size,height=self.size)
            if random.random() < 0.5:
                if self.size == 1440:
                    img, MA_mask, HE_mask, EX_mask, SE_mask = random_scale_and_crop_four(img, MA_mask, HE_mask, EX_mask, SE_mask,target_size=(1440,960))
                else:
                    img, MA_mask, HE_mask, EX_mask, SE_mask = random_scale_and_crop_four(img, MA_mask, HE_mask, EX_mask, SE_mask,target_size=(self.size,self.size))
        else:
            if self.size == 1440:
                img, MA_mask, HE_mask, EX_mask, SE_mask = resize1440_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
            else:
                img, MA_mask, HE_mask, EX_mask, SE_mask = resize_four(img, MA_mask, HE_mask, EX_mask, SE_mask,
                                                                      width=self.size, height=self.size)

        img, MA_mask, HE_mask, EX_mask, SE_mask = normalize_four(img, MA_mask, HE_mask, EX_mask, SE_mask)
        sample['image'] = img
        sample['MA_mask'] = MA_mask if MA_mask is not None else torch.zeros(1)
        sample['HE_mask'] = HE_mask if HE_mask is not None else torch.zeros(1)
        sample['EX_mask'] = EX_mask if EX_mask is not None else torch.zeros(1)
        sample['SE_mask'] = SE_mask if SE_mask is not None else torch.zeros(1)
        return sample

    def __getitem__(self, item):
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)


class IDRIDDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,CLAHE = 2):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = int(size)
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
        if self.CLAHE > 0:
            img_path = img_path.replace('/images','/images_clahe_{}'.format(self.CLAHE))
            img = Image.open(img_path)
        else:
            img = Image.open(img_path)

        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path)

        if self.mode == 'semi_train':
            if random.random() < 0.5:
                img, mask = hflip(img, mask)
            if random.random() < 0.5:
                img, mask = vflip(img, mask)
            if self.size == 1440:
                img, mask = resize1440(img, mask)
                if random.random() < 0.5:
                    img, mask = random_scale_and_crop(img,mask,target_size=(1440,960))
            else:
                img, mask = resize(img, mask, self.size,self.size)
                if random.random() < 0.5:
                    img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size))
            if random.random() < 0.5:
                img, mask = random_rotate(img, mask,max_rotation_angle=30)
            if random.random() < 0.5:
                img, mask = random_translate(img, mask)

        else:
            if self.size == 1440:
                img, mask = resize1440(img, mask)
            else:
                img, mask = resize(img, mask, self.size,self.size)
        img, mask = normalize(img, mask, mean=MEAN_RGB, std=STDDEV_RGB)
        mask[ mask > 4] = 4
        if 'E-ophtha' in mask_path:
            mask[mask == 1] = 3
            mask[mask == 2] = 1
        return {'image': img, 'label': mask,'id':id.split(' ')[1]}

    def __getitem__(self, item):
        import cv2
        cv2.setNumThreads(0)
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)

class DDRDataset(Dataset):
    def __init__(self,name, root, mode, size,
                 id_path=None,CLAHE = 2):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = int(size)
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
        if self.CLAHE > 0:
            img_path = img_path.replace('/images','/images_clahe_{}'.format(self.CLAHE))
            img = Image.open(img_path)
        else:
            img = Image.open(img_path)

        mask_path = os.path.join(self.root, id.split(' ')[1])
        mask = Image.open(mask_path)

        if self.mode == 'semi_train':
            if random.random() < 0.5:
                img, mask = hflip(img, mask)
            if random.random() < 0.5:
                img, mask = vflip(img, mask)
            if self.size == 1440:
                img, mask = resize1440(img, mask)
                if random.random() < 0.5:
                    img, mask = random_scale_and_crop(img,mask,target_size=(1440,960))
            else:
                img, mask = resize(img, mask, self.size,self.size)
                if random.random() < 0.5:
                    img, mask = random_scale_and_crop(img, mask, target_size=(self.size, self.size))
            if random.random() < 0.5:
                img, mask = random_rotate(img, mask,max_rotation_angle=30)
            if random.random() < 0.5:
                img, mask = random_translate(img, mask)

        else:
            if self.size == 1440:
                img, mask = resize1440(img, mask)
            else:
                img, mask = resize(img, mask, self.size,self.size)

        img, mask = normalize(img, mask, mean=MEAN_RGB, std=STDDEV_RGB)

        return {'image': img, 'label': mask}

    def __getitem__(self, item):
        import cv2
        cv2.setNumThreads(0)
        sample = self.get_item_nor(item)

        return sample

    def __len__(self):
        return len(self.ids)





