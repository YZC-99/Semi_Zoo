import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchmetrics


def gt2boundary(gt, ignore_label=-1):  # gt NHW
    gt_ud = gt[:, 1:, :] - gt[:, :-1, :]  # NHW
    gt_lr = gt[:, :, 1:] - gt[:, :, :-1]
    gt_ud = torch.nn.functional.pad(gt_ud, [0, 0, 0, 1, 0, 0], mode='constant', value=0) != 0
    gt_lr = torch.nn.functional.pad(gt_lr, [0, 1, 0, 0, 0, 0], mode='constant', value=0) != 0
    gt_combine = gt_lr + gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt == ignore_label)

    return gt_combine > 0

def mask_to_boundary(mask, boundary_size=4, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((boundary_size, boundary_size), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    return mask - mask_erode


class BoundaryIoU(torchmetrics.Metric):
    def __init__(
            self,
            num_classes,
            task,
            threshold=0.5,
            boundary_size=4,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_classes = num_classes
        self.threshold = threshold
        self.boundary_size = boundary_size
        self.iou = torchmetrics.JaccardIndex(num_classes=2, task=task)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #         print("--------------")
        #         print(target.shape)

        target = target.cpu().detach().numpy().astype("uint8")
        b, h, w = target.shape
        preds = (preds > self.threshold).cpu().detach().numpy().astype("uint8")
        #         print("--------------")
        #         print(preds.shape)
        for i in range(b):
            #             for j in range(c):
            boundary_target = mask_to_boundary(
                target[i], boundary_size=self.boundary_size,
            )
            boundary_target = (
                torch.Tensor(boundary_target).int().to(self.iou.device)
            )
            boundary_preds = mask_to_boundary(
                preds[i], boundary_size=self.boundary_size,
            )
            boundary_preds = torch.Tensor(boundary_preds).int().to(self.iou.device)
            self.iou(boundary_preds, boundary_target)

    def compute(self):
        res = self.iou.compute()
        self.iou.reset()
        return res

    def reset(self):
        self.iou.reset()