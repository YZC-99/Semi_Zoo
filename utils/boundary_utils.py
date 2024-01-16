import torch
from PIL import Image
import numpy as np

def gt2boundary_tensor(gt, ignore_label=-1,boundary_width = 5):  # gt NHW
    gt_ud = gt[:, boundary_width:, :] - gt[:, :-boundary_width, :]  # NHW
    gt_lr = gt[:, :, boundary_width:] - gt[:, :, :-boundary_width]
    gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, boundary_width, 0, 0], mode='constant', value=0) != 0
    gt_lr = torch.nn.functional.pad(gt_lr, [0, boundary_width, 0, 0, 0, 0], mode='constant', value=0) != 0
    gt_combine = gt_lr + gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt == ignore_label)

    return gt_combine > 0


def gt2boundary_numpy(gt, ignore_label=-1,boundary_width = 5):  # gt HW
    gt_ud = gt[boundary_width:, :] - gt[:-boundary_width, :]  # HW
    gt_lr = gt[:, boundary_width:] - gt[:, :-boundary_width]
    gt_ud = np.pad(gt_ud, ((0, boundary_width), (0, 0)), mode='constant', constant_values=0) != 0
    gt_lr = np.pad(gt_lr, ((0, 0), (boundary_width, 0)), mode='constant', constant_values=0) != 0
    gt_combine = gt_lr + gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt == ignore_label)

    return gt_combine > 0


if __name__ == '__main__':
    mask_arr = np.array(Image.open('drishtiGS_002.png'))
    mask_arr[mask_arr == 1] = 0
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
    mask_boundary_tensor = gt2boundary_tensor(mask_tensor)
    mask_boundary_arr = mask_boundary_tensor.squeeze().numpy()
    mask_boundary_arr = mask_boundary_arr.astype(np.uint8)
    mask_boundary = Image.fromarray(mask_boundary_arr * 125)
    mask_boundary.save('boundary_mask.png')