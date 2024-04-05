from tqdm import tqdm
from dataloader.fundus import SemiDataset, IDRIDDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
import os
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

def apply_mask(image, mask,t=False):
    # 创建一个与原图相同大小的纯色图像用于颜色填充
    color_map = {
        1: [255, 0, 0],  # 红色
        2: [0, 255, 0],  # 绿色
        3: [0, 0, 255],  # 蓝色
        4: [255, 255, 0]  # 黄色
    }
    # 将PIL图像转换为Numpy数组以便操作
    image_np = np.array(image)

    mask_np = np.array(mask)
    mask_np[mask_np > 0] = 1
    if t:
        mask_np[mask_np == 1] = 3
        mask_np[mask_np == 2] = 1

    # 对于mask中的每个唯一值，替换图像中对应的像素颜色
    for value, color in color_map.items():
        image_np[mask_np == value] = color

    # 将修改后的Numpy数组转换回PIL图像
    return Image.fromarray(image_np)

if __name__ == '__main__':

    image_path = "D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/IDRiD_Images/IDRiD_55.jpg"
    mask_path = "D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/crop_IDRID/GTs/results/IDRiD_55_fuse.png"
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # 将图像大小调整到和mask一致
    # image = image.resize(mask.size, Image.Resampling.LANCZOS)
    mask = mask.resize(image.size, Image.Resampling.LANCZOS)

    result_image = apply_mask(image, mask)

    # 保存处理后的图像
    result_image_path = "obj-55.jpg"
    result_image.save(result_image_path)

