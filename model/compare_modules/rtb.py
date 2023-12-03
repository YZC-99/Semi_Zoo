"""
RTNet: Relation Transformer Network for Diabetic Retinopathy Multi-lesion Segmentation
Relation Transformer Block
"""
import torch
from torch import nn
import torch.nn.functional as F


class RTB(nn.Module):
    def __init__(self,in_channels):
        super(RTB, self).__init__()
        self.l_conv_Qs = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)
        self.l_conv_Ks = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)
        self.l_conv_Vs = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)

        self.l_conv_Qc = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)
        self.v_conv_Kc = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)
        self.v_conv_Vc = nn.Conv2d(kernel_size=3,in_channels=in_channels,out_channels=in_channels,stride=1,padding=1)

        self.Gs_conv = nn.Conv2d(kernel_size=1,in_channels=in_channels,out_channels=in_channels)
        self.Gc_conv = nn.Conv2d(kernel_size=1,in_channels=in_channels,out_channels=in_channels)

        self.fuse_conv = nn.Conv2d(kernel_size=1,in_channels=in_channels * 2,out_channels=in_channels)
    def forward(self, Fl, Fv):
        b,c,h,w = Fl.shape
        Qs = self.l_conv_Qs(Fl).reshape(b,-1,h * w)
        Ks = self.l_conv_Ks(Fl).reshape(b,-1,h * w)
        Vs = self.l_conv_Vs(Fl).reshape(b,-1,h * w)

        #Qs 与 Ks转置后相乘
        multi_s = torch.einsum('ijk,ijq->ikq',Qs,Ks)
        multi_s = torch.softmax(multi_s,dim=1)
        #
        Gs = torch.einsum('bik,bjk->bij',multi_s,Vs)
        Gs = Gs.reshape(b,c,h,w)
        Gs = self.Gs_conv(Gs)
        Gs = Gs + Fl

        #---------------

        Qs = self.l_conv_Qc(Fl).reshape(b,-1,h * w)
        Kc = self.v_conv_Kc(Fv).reshape(b,-1,h * w)
        Vc = self.v_conv_Vc(Fv).reshape(b,-1,h * w)



        #Qc 与 Kc转置后相乘
        multi_c =  torch.einsum('ijk,ijq->ikq',Qs,Kc)
        multi_c = torch.softmax(multi_c,dim=1)
        #
        Gc = torch.einsum('bik,bjk->bij',multi_c,Vc)
        Gc = Gc.reshape(b,c,h,w)
        Gc = self.Gs_conv(Gc)
        Gc = Gc + Fv

        out = self.fuse_conv(torch.cat([Gs,Gc],dim=1))
        return out

if __name__ == '__main__':
    data = torch.randn(4,64,128,128)
    net = RTB(in_channels=64)
    out = net(data,data) # (4, 128, 128, 128)
    print(out.shape)


