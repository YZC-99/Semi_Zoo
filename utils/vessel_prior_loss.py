import torch
import torch.nn as nn
import torch.nn.functional as F

class KinkLoss(nn.Module):
    def __init__(self):
        super(KinkLoss, self).__init__()


    def kink_oc_mse(self, features, odoc_mask, kink_mask):


    def forward(self, features, odoc_mask, kink_mask):
        """
        :param features: 语义分割模型的logits输出
        :param odoc_mask: 包含0，1，2的标签
        :param kink_mask: 只包含0和1的标签
        :return: CE Loss
        """

        # 获取 kink_mask 中值为 1 的位置的索引
        kink_indices = torch.nonzero(kink_mask == 1, as_tuple=False)

        # 从 features 和 odoc_mask 中提取对应索引位置的值
        features_at_kink = features[kink_indices[:, 0], :, kink_indices[:, 1], kink_indices[:, 2]]
        odoc_mask_at_kink = odoc_mask[kink_indices[:, 0], kink_indices[:, 1], kink_indices[:, 2]]

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(features_at_kink, odoc_mask_at_kink)

        return ce_loss
