import torch
import torch.nn as nn
from torch.distributions import normal
import torch.nn.functional as F

class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4):
        super(BlvLoss, self).__init__()
        cls_num_list = torch.tensor(cls_num_list)
        cls_list = torch.tensor(cls_num_list, dtype=torch.float)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_num_list)) - frequency_list
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = 'BlvLoss'

    def forward(self, pred, target):
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1)
        viariation = viariation.to(pred.device)
        self.frequency_list = self.frequency_list.to(pred.device)
        pred = pred + (viariation.abs().permute(0, 2, 3, 1) / self.frequency_list.max() * self.frequency_list).permute(0, 3, 1, 2)
        loss = F.cross_entropy(pred, target, reduction='none')
        return loss