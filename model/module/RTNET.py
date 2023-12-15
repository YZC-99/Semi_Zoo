"""
RTNet: Relation Transformer Network for Diabetic Retinopathy Multi-lesion Segmentation
"""
import torch
import torch.nn as nn
from model.module.gap import GlobalAvgPool2D

class Global_Transformer_Block(nn.Module):
    def __init__(self,in_channels):
        super(Global_Transformer_Block, self).__init__()

        self.proj_Q = nn.Conv2d(in_channels,in_channels,3,1,1)
        self.proj_Q_pool = GlobalAvgPool2D()
        self.proj_K = nn.Conv2d(in_channels,in_channels,3,1,1)
        self.proj_V = nn.Conv2d(in_channels,in_channels,3,1,1)

        self.conv1 = nn.Conv2d(in_channels,1,1,1)
    def forward(self,x):
        Q = self.proj_Q(x)
        Q = self.proj_Q_pool(Q)
        K = self.proj_K(x)
        b,c,h,w = K.size()
        K = torch.reshape(K,[b,c,-1])
        V = self.proj_V(x)
        V = torch.reshape(V,[b,c,-1])
        result_QK = torch.einsum("bil,bi->bl", K, Q.squeeze(-1).squeeze(-1))
        result_QK = torch.softmax(result_QK,dim=1)

        result_QKV = torch.einsum('bl,bcl->bc',result_QK,V)
        result_QKV = result_QKV.unsqueeze(-1).unsqueeze(-1)
        result_QKV = self.conv1(result_QKV)

        out = result_QKV + x
        return out

if __name__ == '__main__':
    data = torch.randn(4,32,256,256)
    GTB = Global_Transformer_Block(32)
    out = GTB(data)
    print(out)