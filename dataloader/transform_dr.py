import numpy as np
from PIL import Image, ImageOps, ImageFilter,ImageEnhance
import random
import torch
from torchvision import transforms
from dataloader.boundary_utils import class2one_hot,one_hot2dist



def vflip_four(img, mask1=None,mask2=None,mask3=None,mask4=None):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if mask1 is not None:
        mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
    if mask2 is not None:
        mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
    if mask3 is not None:
        mask3 = mask3.transpose(Image.FLIP_TOP_BOTTOM)
    if mask4 is not None:
        mask4 = mask4.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask1,mask2,mask3,mask4

def hflip_four(img, mask1=None,mask2=None,mask3=None,mask4=None):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if mask1 is not None:
        mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
    if mask2 is not None:
        mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
    if mask3 is not None:
        mask3 = mask3.transpose(Image.FLIP_LEFT_RIGHT)
    if mask4 is not None:
        mask4 = mask4.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask1,mask2,mask3,mask4


def random_scale_four(img, mask1=None,mask2=None,mask3=None,mask4=None, min_scale=0.8, max_scale=1.2 ):
    w_scale_factor = random.uniform(min_scale, max_scale)
    h_scale_factor = random.uniform(min_scale, max_scale)
    new_width = int(img.width * w_scale_factor)
    new_height = int(img.height * h_scale_factor)

    img = img.resize((new_width, new_height), Image.BILINEAR)
    if mask1 is not None:
        mask1 = mask1.resize((new_width, new_height), Image.NEAREST)
    if mask2 is not None:
        mask2 = mask2.resize((new_width, new_height), Image.NEAREST)
    if mask3 is not None:
        mask3 = mask3.resize((new_width, new_height), Image.NEAREST)
    if mask4 is not None:
        mask4 = mask4.resize((new_width, new_height), Image.NEAREST)
    return img, mask1,mask2,mask3,mask4



def random_scale_and_crop_four(img, mask1=None,mask2=None,mask3=None,mask4=None, target_size=(256, 256), min_scale=0.8, max_scale=1.2):
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
    if mask1 is not None:
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
    if mask2 is not None:
        mask2 = mask2.resize((ow, oh), Image.NEAREST)
    if mask3 is not None:
        mask3 = mask3.resize((ow, oh), Image.NEAREST)
    if mask4 is not None:
        mask4 = mask4.resize((ow, oh), Image.NEAREST)

    # pad crop
    if short_size < target_size[0]:
        padh = target_size[1] - oh if oh < target_size[1] else 0
        padw = target_size[0] - ow if ow < target_size[0] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        if mask1 is not None:
            mask1 = ImageOps.expand(mask1, border=(0, 0, padw, padh), fill=0)
        if mask2 is not None:
            mask2 = ImageOps.expand(mask2, border=(0, 0, padw, padh), fill=0)
        if mask3 is not None:
            mask3 = ImageOps.expand(mask3, border=(0, 0, padw, padh), fill=0)
        if mask4 is not None:
            mask4 = ImageOps.expand(mask4, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - target_size[0])
    y1 = random.randint(0, h - target_size[1])
    img = img.crop((x1, y1, x1 + target_size[0], y1 + target_size[1]))
    if mask1 is not None:
        mask1 = mask1.crop((x1, y1, x1 + target_size[0], y1 + target_size[1]))
    if mask2 is not None:
        mask2 = mask2.crop((x1, y1, x1 + target_size[0], y1 + target_size[1]))
    if mask3 is not None:
        mask3 = mask3.crop((x1, y1, x1 + target_size[0], y1 + target_size[1]))
    if mask4 is not None:
        mask4 = mask4.crop((x1, y1, x1 + target_size[0], y1 + target_size[1]))
    return img, mask1,mask2,mask3,mask4



def random_rotate_four(img,  mask1=None,mask2=None,mask3=None,mask4=None, max_rotation_angle=90):
    rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
    img = img.rotate(rotation_angle, resample=Image.BILINEAR)
    if mask1 is not None:
        mask1 = mask1.rotate(rotation_angle, resample=Image.NEAREST)
    if mask2 is not None:
        mask2 = mask2.rotate(rotation_angle, resample=Image.NEAREST)
    if mask3 is not None:
        mask3 = mask3.rotate(rotation_angle, resample=Image.NEAREST)
    if mask4 is not None:
        mask4 = mask4.rotate(rotation_angle, resample=Image.NEAREST)
    return img, mask1,mask2,mask3,mask4

def random_translate_four(img,   mask1=None,mask2=None,mask3=None,mask4=None, max_translate_percent=0.15):
    img_width, img_height = img.size
    translate_x = random.uniform(-max_translate_percent, max_translate_percent) * img_width
    translate_y = random.uniform(-max_translate_percent, max_translate_percent) * img_height

    img = img.transform(
        img.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
    )
    if mask1 is not None:
        mask1 = mask1.transform(
            mask1.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )
    if mask2 is not None:
        mask2 = mask2.transform(
            mask2.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )
    if mask3 is not None:
        mask3 = mask3.transform(
            mask3.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )
    if mask4 is not None:
        mask4 = mask4.transform(
            mask4.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y)
        )
    return img, mask1,mask2,mask3,mask4




def normalize_four(img,   mask1=None,mask2=None,mask3=None,mask4=None,mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0]):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])(img)
    if mask1 is not None:
        mask1 = torch.from_numpy(np.array(mask1)).long()
    if mask2 is not None:
        mask2 = torch.from_numpy(np.array(mask2)).long()
    if mask3 is not None:
        mask3 = torch.from_numpy(np.array(mask3)).long()
    if mask4 is not None:
        mask4 = torch.from_numpy(np.array(mask4)).long()
    return img, mask1,mask2,mask3,mask4

def resize_four(img, mask1=None,mask2=None,mask3=None,mask4=None,width=512,height=512):
    img = img.resize((width, height), Image.BILINEAR)
    if mask1 is not None:
        mask1 = mask1.resize((width, height), Image.NEAREST)
    if mask2 is not None:
        mask2 = mask2.resize((width, height), Image.NEAREST)
    if mask3 is not None:
        mask3 = mask3.resize((width, height), Image.NEAREST)
    if mask4 is not None:
        mask4 = mask4.resize((width, height), Image.NEAREST)
    return img, mask1,mask2,mask3,mask4

def resize1440_four(img, mask1=None,mask2=None,mask3=None,mask4=None):
    img = img.resize((1440,960), Image.BILINEAR)
    if mask1 is not None:
        mask1 = mask1.resize((1440,960), Image.NEAREST)
    if mask2 is not None:
        mask2 = mask2.resize((1440,960), Image.NEAREST)
    if mask3 is not None:
        mask3 = mask3.resize((1440,960), Image.NEAREST)
    if mask4 is not None:
        mask4 = mask4.resize((1440,960), Image.NEAREST)
    return img, mask1,mask2,mask3,mask4






