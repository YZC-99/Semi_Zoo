import torch
import torch.nn as nn
import torch.nn.functional as F

class KinkLoss(nn.Module):
    def __init__(self):
        super(KinkLoss, self).__init__()


    def oc_features_center(self, features, odoc_mask):
        # 获得oc区域logits的索引
        oc_indices = torch.nonzero(odoc_mask == 2, as_tuple=False)
        features_at_oc = features[oc_indices[:,0],:,oc_indices[:,1],oc_indices[:,2]]
        # features_at_oc = features_at_oc.view(features.shape[0],-1)
        oc_features_center = torch.mean(features_at_oc,dim=0,keepdim=True)
        return oc_features_center

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
        # odoc_mask_at_kink = odoc_mask[kink_indices[:, 0], kink_indices[:, 1], kink_indices[:, 2]]
        oc_features_center = self.oc_features_center(features, odoc_mask)
        oc_features_center = oc_features_center.tile((features_at_kink.shape[0],1))
        # 计算交叉熵损失
        mse_loss = F.mse_loss(oc_features_center, features_at_kink)
        return mse_loss

if __name__ == '__main__':


    features = torch.randn(4,256,128,128)
    odoc_mask = torch.ones(4,1,128,128) * 2
    vessel_mask = torch.ones_like(odoc_mask)
    kink_loss = KinkLoss()
    loss = kink_loss(features,odoc_mask,vessel_mask)
    print(loss)
