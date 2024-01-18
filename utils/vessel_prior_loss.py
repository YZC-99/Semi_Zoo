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


    def mse(self, features, odoc_mask, kink_mask):


        # 获取 kink_mask 中值为 1 的位置的索引
        kink_indices = torch.nonzero(kink_mask == 1, as_tuple=False)

        features_at_kink = features[kink_indices[:, 0], :, kink_indices[:, 1], kink_indices[:, 2]]

        oc_features_center = self.oc_features_center(features, odoc_mask)
        oc_features_center = oc_features_center.tile((features_at_kink.shape[0],1))

        mse_loss = F.mse_loss(oc_features_center.detach(), features_at_kink)
        F.triplet_margin_loss()
        return mse_loss


    def cos_mse(self, features, odoc_mask, kink_mask):

        # 获取 kink_mask 中值为 1 的位置的索引
        kink_indices = torch.nonzero(kink_mask == 1, as_tuple=False)


        # 从 features 和 odoc_mask 中提取对应索引位置的值
        features_at_kink = features[kink_indices[:, 0], :, kink_indices[:, 1], kink_indices[:, 2]]
        odoc_mask_at_kink = odoc_mask[kink_indices[:, 0], kink_indices[:, 1], kink_indices[:, 2]]

        oc_features_center = self.oc_features_center(features, odoc_mask)
        oc_features_center = oc_features_center.tile((features_at_kink.shape[0],1))

        simi = F.cosine_similarity(oc_features_center.detach(), features_at_kink)
        # 计算交叉熵损失
        mse_loss = F.mse_loss(simi, odoc_mask_at_kink.float())


        if torch.isnan(mse_loss):
            return torch.zeros(1,device=features.device)
        return mse_loss


    def forward(self, features, odoc_mask, kink_mask,mse_type='mse'):
        if mse_type == 'mse':
            mse_loss = self.mse(features, odoc_mask, kink_mask)
        else:
            mse_loss = self.cos_mse(features, odoc_mask, kink_mask)


        if torch.isnan(mse_loss):
            return torch.zeros(1,device=features.device)
        return mse_loss

if __name__ == '__main__':


    features = torch.randn(4,256,128,128)
    odoc_mask = torch.ones(4,128,128) * 2
    vessel_mask = torch.ones_like(odoc_mask)
    kink_loss = KinkLoss()
    loss = kink_loss(features,odoc_mask,vessel_mask,'cos-mse')
    print(loss)
