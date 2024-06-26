"""
Selective Kernel Networks
"""
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict
from model.module.external_attentions.cbam import CBAMBlock,SpatialAttention

class DAM(nn.Module):
    def __init__(self,in_channels=512):
        super(DAM, self).__init__()

        self.dam1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                  dilation=1, bias=False)
        self.dam2 = nn.Conv2d(in_channels, in_channels, 3, padding=2,
                  dilation=2, bias=False)
        self.dam4 = nn.Conv2d(in_channels, in_channels, 3, padding=4,
                  dilation=4, bias=False)
        self.dam8 = nn.Conv2d(in_channels, in_channels, 3, padding=8,
                  dilation=8, bias=False)

        self.block = nn.Sequential(nn.Conv2d(in_channels*5, in_channels, 3, padding=1,
                                    bias=False),
                          nn.BatchNorm2d(in_channels),
                          nn.ReLU(True))

    def forward(self,x):
        d1 = self.dam1(x)
        d2 = self.dam2(d1)
        d4 = self.dam4(d2)
        d8 = self.dam8(d4)

        _d = torch.cat([x,d1,d2,d4,d8],dim=1)

        _c = self.block(_d)
        return _c



class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weights=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weights=self.softmax(attention_weights)#k,bs,channel,1,1

        ### fuse
        V=(attention_weights*feats).sum(0)
        return V

class SK_Dali_Attention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.convs.append(DAM(channel))
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)+1):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weights=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weights=self.softmax(attention_weights)#k,bs,channel,1,1

        ### fuse
        V=(attention_weights*feats).sum(0)
        return V




class SK_Spatial_Attention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )

        self.fss=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fss.append(SpatialAttention())


    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w


        ### calculate spatial attention weight
        weights=[]
        for fs in self.fss:
            weight=fs(U)
            weights.append(weight) #bs,1,h,w
        attention_weights=torch.stack(weights,0)#k,bs,1,h,w
        ### fuse
        V=(attention_weights).sum(0)
        return V


class SK_add_Spatial_Attention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5],reduction=16,group=1,L=32):
        super().__init__()
        self.org_sk = SKAttention(channel)
        self.spatial = SK_Spatial_Attention(channel)

    def forward(self, x):
        return self.org_sk(x) + self.spatial(x)

class SK_add_CBAM_Attention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5],reduction=16,group=1,L=32):
        super().__init__()
        self.org_sk = SKAttention(channel)
        self.cbam = CBAMBlock(channel)

    def forward(self, x):
        return self.org_sk(x) + self.cbam(x)



class SKA_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(SKA_Module, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            sa_layer = 'ska_layer{}'.format(idx)
            sa_layer_module = SKAttention( in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)

    def forward(self, fpn_features):
        for idx,(feature,sa) in enumerate(zip(fpn_features,self.sa_layers)):
            feature_out = getattr(self,sa)(feature)
            fpn_features[idx] = feature_out
        return fpn_features

class SKA_Dali_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(SKA_Dali_Module, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            sa_layer = 'ska_layer{}'.format(idx)
            sa_layer_module = SK_Dali_Attention(in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)

    def forward(self, fpn_features):
        for idx,(feature,sa) in enumerate(zip(fpn_features,self.sa_layers)):
            feature_out = getattr(self,sa)(feature)
            fpn_features[idx] = feature_out
        return fpn_features


class SKA_Spatial_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(SKA_Spatial_Module, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            sa_layer = 'ska_layer{}'.format(idx)
            sa_layer_module = SK_add_Spatial_Attention(in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)

    def forward(self, fpn_features):
        for idx,(feature,sa) in enumerate(zip(fpn_features,self.sa_layers)):
            feature_out = getattr(self,sa)(feature)
            fpn_features[idx] = feature_out
        return fpn_features

class SKA_CBAM_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(SKA_CBAM_Module, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            sa_layer = 'ska_layer{}'.format(idx)
            sa_layer_module = SK_add_CBAM_Attention(in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)

    def forward(self, fpn_features):
        for idx,(feature,sa) in enumerate(zip(fpn_features,self.sa_layers)):
            feature_out = getattr(self,sa)(feature)
            fpn_features[idx] = feature_out
        return fpn_features


if __name__ == '__main__':
    input = torch.randn(4,256,32,32)
    attention = SK_2_CBAM_Attention(256)
    out = attention(input)
    print(out.shape)