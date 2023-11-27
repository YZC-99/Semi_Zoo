import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageEnhance
import random
import torch
from torchvision import transforms
from dataloader.boundary_utils import class2one_hot,one_hot2dist
import cv2



def vflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def random_scale_and_crop(img, mask, target_size=(256, 256), min_scale=0.8, max_scale=1.2,p=0.5):
    if random.random() < p:
        # random scale (short edge)
        short_size = random.randint(int(img.width  * min_scale), int(img.width  * max_scale))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < target_size[0]:
            padh = target_size[0] - oh if oh < target_size[0] else 0
            padw = target_size[0] - ow if ow < target_size[0] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            if mask is not None:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - target_size[0])
        y1 = random.randint(0, h - target_size[0])
        img = img.crop((x1, y1, x1 + target_size[0], y1 + target_size[0]))
        if mask is not None:
            mask = mask.crop((x1, y1, x1 + target_size[0], y1 + target_size[0]))
    return img, mask



def random_rotate(img, mask, p=0.5, max_rotation_angle=90):
    if random.random() < p:
        rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
        img = img.rotate(rotation_angle, resample=Image.BILINEAR, expand=True)
        mask = mask.rotate(rotation_angle, resample=Image.NEAREST, expand=True)

    return img, mask

def random_translate(img, mask, p=0.5, max_translate_percent=0.15):
    if random.random() < p:
        img_width, img_height = img.size
        translate_x = random.uniform(-max_translate_percent, max_translate_percent) * img_width
        translate_y = random.uniform(-max_translate_percent, max_translate_percent) * img_height

        img = img.transform(
            img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

        mask = mask.transform(
            mask.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )

    return img, mask




def normalize(img, mask=None,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # transforms.Normalize([151.818,74.596,23.749], [28.438,15.263 ,5.225]),
        # Mean: [151.81788834  74.5958448   23.74884842]
        # Std: [28.43763701 15.26303392  5.22472751]
        transforms.Normalize((0.0,), (1.0,))
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
    return img,mask

def resize(img, mask,size):
    img = img.resize((size, size), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((size, size), Image.NEAREST)
    return img, mask







