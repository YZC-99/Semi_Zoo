from scipy import ndimage
from PIL import Image
import numpy as np
from torchvision import transforms
import torch


def resize_np(sample,output_size=256):
    output_size = (output_size,output_size)
    image, label = sample['image'], sample['label']
    # Get the current size of the image and label
    current_size = image.shape[:2]

    # Calculate the zoom factors for both dimensions
    zoom_factors = [new_size / current_size[i] for i, new_size in enumerate(output_size)]
    zoom_factors.append(1.0)
    # Resize the image using ndimage.zoom
    image = ndimage.zoom(image, zoom_factors, order=2)

    # Resize the label using ndimage.zoom
    label = ndimage.zoom(label, zoom_factors[:2], order=0)  # Use order=0 for nearest-neighbor interpolation

    # Return the resized image and label as a dictionary
    return {'image': image, 'label': label}

def random_flip_np(sample):
    image, label = sample['image'], sample['label']

    axis = np.random.randint(0,2)
    image = np.flip(image,axis=axis).copy()
    label = np.flip(label,axis=axis).copy()

    return {'image': image, 'label': label}

def random_rotate_np(sample):
    image, label= sample['image'], sample['label']
    angle = np.random.randint(-20,20)
    image = ndimage.rotate(image,angle,order=1,reshape=False)
    label = ndimage.rotate(label,angle,order=0,reshape=False)
    return {'image': image, 'label': label}


def random_scale_crop_np(sample,output_size=256,min_scale = 0.8,max_scale=1.2):
    zoom_factors = [np.random.uniform(min_scale,max_scale),np.random.uniform(min_scale,max_scale),1.0]

    image, label= sample['image'], sample['label']
    # Resize the image using ndimage.zoom
    resized_image = ndimage.zoom(image, zoom_factors, order=2)
    # Resize the label using ndimage.zoom
    resized_label = ndimage.zoom(label, zoom_factors[:2], order=0)  # Use order=0 for nearest-neighbor interpolation

    # 如果resized_image的高或者宽小于output_size，那么就给resized_image和resized_label相应的高或者宽用0填充
    resized_height, resized_width = resized_image.shape[:2]

    # 计算需要填充的高度和宽度
    pad_height = max(0, output_size - resized_height)
    pad_width = max(0, output_size - resized_width)

    # 计算填充的上下左右边距
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # 在上下左右进行填充
    padded_image = np.pad(resized_image, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0)
    padded_label = np.pad(resized_label, ((top, bottom), (left, right)), mode='constant', constant_values=0)

    # 按照output_size，在resized_image和resized_label上进行中心裁剪
    # 中心裁剪到指定大小
    cropped_image = padded_image[:output_size, :output_size, :]
    cropped_label = padded_label[:output_size, :output_size]

    return {'image': cropped_image, 'label': cropped_label}


def normalize_np(sample,mean=(0.0,),std=(1.0,)):
    image, label = sample['image'], sample['label']
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])(image)
    label = torch.from_numpy(label).long()

    return {'image': image, 'label': label}