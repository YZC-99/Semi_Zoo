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
    if t:
        mask_np[mask_np == 1] = 3
        mask_np[mask_np == 2] = 1

    # 对于mask中的每个唯一值，替换图像中对应的像素颜色
    for value, color in color_map.items():
        image_np[mask_np == value] = color

    # 将修改后的Numpy数组转换回PIL图像
    return Image.fromarray(image_np)

if __name__ == '__main__':


    # init dataset
    # images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/数据集/实验结果/exp_refer_2d_dr/IDRiD_Images'
    # dataset_name = 'crop_IDRID'
    # image_size = 1440

    # images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/数据集/实验结果/exp_refer_2d_dr/DDR_Images'
    # dataset_name = 'DDR'
    # image_size = 1024

    # images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/数据集/实验结果/exp_refer_2d_dr/E-ophtha_images'
    # dataset_name = 'E-ophtha'
    # image_size = 1024

    # icic
    # images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/IDRiD_Images'
    # dataset_name = 'crop_IDRID'
    # image_size = 1440

    # images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/DDR_Images'
    # dataset_name = 'DDR'
    # image_size = 1024


    images_root_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/E-ophtha_images'
    dataset_name = 'E-ophtha'
    image_size = 1024


    pred_root_base = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/DR_分割/ICIC2024/实验结果/' + dataset_name

    for m in os.listdir(pred_root_base):
        model = m


        pred_base = os.path.join(pred_root_base,model, "results")
        cover_base = os.path.join(pred_root_base, model, "cover")
        os.makedirs(cover_base, exist_ok=True)


        for i in os.listdir(images_root_path):
            image_path = os.path.join(images_root_path,i)
            # mask_name = i.replace('.jpg','_fuse.png')



            # mask_name = i.replace('.jpg','.png')
            mask_name = i.replace('.jpg','_merged.png').replace('.JPG','_merged.png')
            # Find the corresponding mask image
            mask_path = os.path.join(pred_base,mask_name)

            """
           mask是一个灰度图像，与image_path一样
           1、将image的大小调整到和mask一致
           2、实现mask掩盖在原始图上的效果。我希望mask上像素值为1对应image的部分被替代为红色，2为绿色，3为蓝色，4为黄色
           3、被替代后的图片保存到下面这个路径：cover_base，图片名字为image_id
           """
            if not os.path.exists(mask_path):
                print("{} 不存在".format(mask_path))
                continue  # 如果对应的mask不存在，则跳过

            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # 将图像大小调整到和mask一致
            # image = image.resize(mask.size, Image.Resampling.LANCZOS)
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)


            # 应用mask
            if dataset_name == 'E-ophtha' and m == 'GTs':
                result_image = apply_mask(image, mask,True)
            else:
                result_image = apply_mask(image, mask)

            # 保存处理后的图像
            result_image_path = os.path.join(cover_base, i)
            result_image.save(result_image_path)
        print("{} 已完成".format(m))
    print("All images have been processed and saved.")

