"""
E:\Deep_Learning_DATABASE\fundus_images\E-Ophtha是一个数据集的路径
下面有
- e_optha_EX
    - Annotation_EX
    - EX
- e_optha_MA
    - Annotation_MA
    - MA

这里面以e_optha_EX为例，e_optha_MA的内容与e_optha_EX一样，
其中
 Annotation_EX和  EX
下面的文件夹名字是一一对应的，一个是注释文件夹，一个是原文件夹。
原文件夹里面的照片都能在注释文件夹里面找到同名的照片，但后缀可能不一样。
其中图片的注释形式是一个二值图片，其中像素值为0的是背景，而像素值为255的是前景
 Annotation_MA和  MA文件的也与 Annotation_EX和  EX的一致

现在已知的是，MA和EX的原文件夹下面的图片有一部分是同名的，那么这部分就意味着有多个注释
我需要完成的是
1、创建一个文件夹为e_optha_EX_MA，并在其下面创建images和annotations文件夹
2、寻找EX原文件夹下面的图片与MA原文件夹下面的图片同名的部分，同名的部分是一个文件夹，需要操作的是同名文件夹下面的图片文件
3、将该部分复制到e_optha_EX_MA下的images，同样Annotation_EX与Annotation_MA下面也有同名的文件夹，同名文件夹下面的就是注释图片，令一张图片里面EX的注释像素值为
1，而MA的注释像素值为2，其他的为0，并且利用PIL里面Image对象的伪彩色保存到e_optha_EX_MA下的annotations。1对应红色，2对应绿色

"""
import os
from PIL import Image
import numpy as np
import shutil


def merge_annotations(ex_annotation_path, ma_annotation_path, output_path):
    ex_img = Image.open(ex_annotation_path).convert('L')
    ma_img = Image.open(ma_annotation_path).convert('L')
    ex_array = np.array(ex_img)
    ma_array = np.array(ma_img)

    merged_array = np.zeros(ex_array.shape, dtype=np.uint8)
    merged_array[ex_array == 255] = 1  # EX的注释设为1
    merged_array[ma_array == 255] = 2  # MA的注释设为2

    merged_img = Image.fromarray(merged_array, mode='P')
    merged_img.putpalette([
        0, 0, 0,  # 背景色
        255, 0, 0,  # 1对应红色
        0, 255, 0,  # 2对应绿色
    ])
    merged_img.save(output_path)


def process_dataset(base_path, output_path):
    ex_path = os.path.join(base_path, 'e_optha_EX')
    ma_path = os.path.join(base_path, 'e_optha_MA')

    images_output = os.path.join(output_path, 'images')
    annotations_output = os.path.join(output_path, 'annotations')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(annotations_output, exist_ok=True)

    ex_suffix = '_EX.png'
    ma_suffix = '.png'

    common_folders = set(os.listdir(os.path.join(ex_path, 'EX'))).intersection(os.listdir(os.path.join(ma_path, 'MA')))

    for folder_name in common_folders:
        ex_images_folder = os.path.join(ex_path, 'EX', folder_name)
        ma_images_folder = os.path.join(ma_path, 'MA', folder_name)
        ex_annotation_folder = os.path.join(ex_path, 'Annotation_EX', folder_name)
        ma_annotation_folder = os.path.join(ma_path, 'Annotation_MA', folder_name)

        for image_name in os.listdir(ex_images_folder):
            if image_name in os.listdir(ma_images_folder):
                # 复制图片到目标images文件夹
                shutil.copy2(os.path.join(ex_images_folder, image_name), os.path.join(images_output, image_name))

                image_base = os.path.splitext(image_name)[0]
                ex_annotation_name = image_base + ex_suffix
                ma_annotation_name = image_base + ma_suffix
                ex_annotation_path = os.path.join(ex_annotation_folder, ex_annotation_name)
                ma_annotation_path = os.path.join(ma_annotation_folder, ma_annotation_name)

                if os.path.exists(ex_annotation_path) and os.path.exists(ma_annotation_path):
                    merged_annotation_path = os.path.join(annotations_output, f"{image_base}_merged.png")
                    merge_annotations(ex_annotation_path, ma_annotation_path, merged_annotation_path)
                else:
                    print(
                        f"Missing annotation for {image_name}. EX exists: {os.path.exists(ex_annotation_path)}, MA exists: {os.path.exists(ma_annotation_path)}")


base_path = 'E:/Deep_Learning_DATABASE/fundus_images/E-Ophtha'
output_path = os.path.join(base_path, 'e_optha_EX_MA')
os.makedirs(output_path, exist_ok=True)

process_dataset(base_path, output_path)
print("处理完成。")
