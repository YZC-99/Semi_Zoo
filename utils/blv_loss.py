import torch
import torch.nn as nn
from torch.distributions import normal
import torch.nn.functional as F
import math
import numpy as np
# import mmcv
#


def cosine_annealing(lower_bound, upper_bound, _t, _t_max):
    return upper_bound + 0.5 * (lower_bound - upper_bound) * (math.cos(math.pi * _t / _t_max) + 1)


#
#
# def get_class_weight(class_weight):
#     """Get class weight for loss function.
#
#     Args:
#         class_weight (list[float] | str | None): If class_weight is a str,
#             take it as a file name and read from it.
#     """
#     if isinstance(class_weight, str):
#         # take it as a file path
#         if class_weight.endswith('.npy'):
#             class_weight = np.load(class_weight)
#         else:
#             # pkl, json or yaml
#             class_weight = mmcv.load(class_weight)
#
#     return class_weight

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class BlvLoss(nn.Module):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name



    def forward(self, pred, target, weight=None, ignore_index=255, avg_factor=None, reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

        if weight is not None:
            weight = weight.float()

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return loss


class Softmaxfocal_BlvLoss(nn.Module):
#cls_nufrequency_list
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(Softmaxfocal_BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def softmaxfocal(self,losses,y_pred, y_true,ignore_index=255,gamma=2.0,normalize=True):
        with torch.no_grad():
            p = y_pred.softmax(dim=1)
            modulating_factor = (1 - p).pow(gamma)
            valid_mask = ~ y_true.eq(ignore_index)
            masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
            modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
            scale = 1.
            if normalize:
                scale = losses.sum() / (losses * modulating_factor).sum()
        return scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    def forward(self, pred, target, weight=None, ignore_index=255, avg_factor=None, reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

        loss = self.softmaxfocal(loss,pred, target)
        return loss


class Annealing_Softmaxfocal_BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(Annealing_Softmaxfocal_BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def annealing_softmaxfocal(self,losses,y_pred, y_true,t, t_max,ignore_index=255,gamma=2.0,annealing_function=cosine_annealing):
        with torch.no_grad():
            p = y_pred.softmax(dim=1)
            modulating_factor = (1 - p).pow(gamma)
            valid_mask = ~ y_true.eq(ignore_index)
            masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
            modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
            normalizer = losses.sum() / (losses * modulating_factor).sum()
            scales = modulating_factor * normalizer
            if t > t_max:
                scale = scales
            else:
                scale = annealing_function(1, scales, t, t_max)
            losses = (losses * scale).sum() / (valid_mask.sum() + p.size(0))
        return losses

    def forward(self, pred, target, t, t_max, ignore_index=255, avg_factor=None, reduction_override=None):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)

        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)

        loss = F.cross_entropy(pred, target, reduction='none',  ignore_index=ignore_index)

        loss = self.annealing_softmaxfocal(loss,pred, target,t, t_max)
        return loss

