import h5py
from sdf import compute_sdf_luoxd,compute_sdf_dual
import numpy as np
from PIL import Image
import os

def gt2boundary(gt, ignore_label=-1, boundary_width=5):
    gt_ud = gt[boundary_width:, :] - gt[:-boundary_width, :]
    gt_lr = gt[:, boundary_width:] - gt[:, :-boundary_width]
    gt_ud = np.pad(gt_ud, ((0, boundary_width), (0, 0)), mode='constant', constant_values=0) != 0
    gt_lr = np.pad(gt_lr, ((0, 0), (0, boundary_width)), mode='constant', constant_values=0) != 0
    gt_combine = gt_lr + gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt == ignore_label)

    return gt_combine > 0

def single_img_mask_2_h5(img_path,mask_path,h5_path,have_mask = True):
    img = Image.open(img_path)
    img_array = np.array(img)

    if not have_mask:
        mask_array = np.zeros([2,2])
    else:
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

    with h5py.File(h5_path,'w') as h5_file:
        # 创建group
        data_group = h5_file.create_group('data')
        data_group.create_dataset('image',data=img_array)
        data_group.create_dataset('label',data=mask_array)

def single_img_mask_contour_sdm_2_h5(img_path,mask_path,h5_path):
    img = Image.open(img_path)
    img_array = np.array(img)

    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    od_mask_array = np.zeros_like(mask_array)
    oc_mask_array = np.zeros_like(mask_array)
    od_mask_array[mask_array > 0] = 1
    oc_mask_array[mask_array > 1] = 1

    od_sdm_luoxd = compute_sdf_luoxd(od_mask_array)
    oc_sdm_luoxd = compute_sdf_luoxd(oc_mask_array)
    sdm_luoxd = np.array([od_sdm_luoxd,oc_sdm_luoxd])

    od_sdm_dual = compute_sdf_dual(od_mask_array)
    oc_sdm_dual = compute_sdf_dual(oc_mask_array)
    sdm_dual = np.array([od_sdm_dual,oc_sdm_dual])

    od_contour = gt2boundary(od_mask_array).astype(np.int8)
    oc_contour = gt2boundary(oc_mask_array).astype(np.int8)
    contours = np.array([od_contour,oc_contour])


    with h5py.File(h5_path,'w') as h5_file:
        # 创建group
        data_group = h5_file.create_group('data')
        data_group.create_dataset('image',data=img_array)
        data_group.create_dataset('label',data=mask_array)
        data_group.create_dataset('sdm_luoxd',data=sdm_luoxd)
        data_group.create_dataset('sdm_dual',data=sdm_dual)
        data_group.create_dataset('contours',data=contours)


def collect_img_mask_2_h5_from_index(data_dir,h5_root,img_mask_index_path,have_mask = True):
    with open(img_mask_index_path,'r') as f:
        all_indexs = f.read().splitlines()
    for line in all_indexs:
        img_name,mask_name = line.split(' ')[0],line.split(' ')[1]
        img_path = os.path.join(data_dir,img_name.replace('/DDR','DDR'))
        mask_path = os.path.join(data_dir,mask_name.replace('/DDR','DDR'))

        h5_dir = img_name.split('/')[0]
        h5_dir = os.path.join(h5_root,h5_dir)
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)

        h5_path = os.path.join(h5_dir,os.path.basename(img_path).split('.')[0] + '.h5')
        single_img_mask_2_h5(img_path,mask_path,h5_path,have_mask)
        # single_img_mask_contour_sdm_2_h5(img_path,mask_path,h5_path)



if __name__ == '__main__':

    data_dir = '/home/gu721/yzc/data/odoc'
    h5_dir = '/home/gu721/yzc/data/odoc/DDR_h5'
    if not os.path.exists(h5_dir):
        os.makedirs(h5_dir)
    img_mask_index_path = '../dataset/DDR/all_index.txt'
    collect_img_mask_2_h5_from_index(data_dir,h5_dir,img_mask_index_path,False)
