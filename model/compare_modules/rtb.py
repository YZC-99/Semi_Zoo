"""
RTNet: Relation Transformer Network for Diabetic Retinopathy Multi-lesion Segmentation
Relation Transformer Block
"""
import torch
from torch import nn
from torch.nn import Softmax
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



def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossRTB(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_channels):
        super(CrissCrossRTB, self).__init__()
        # 用于特征降维
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, Fl, Fv):
        m_batchsize, _, height, width = Fl.size()
        proj_query = self.query_conv(Fv)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(Fl)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(Fl)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to(Fl.device)).view(
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + Fl


class CrissCrossRTB_v2(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_channels):
        super(CrissCrossRTB_v2, self).__init__()
        # 用于特征降维
        self.query_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)

        self.key_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)

        self.value_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.softmax = Softmax(dim=3)
        self.INF = INF

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))



    def forward_1(self, Fl, Fv):
        m_batchsize, _, height, width = Fl.size()
        proj_query = self.query_conv1(Fv)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv1(Fl)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv1(Fl)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to(Fl.device)).view(
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma1 * (out_H + out_W) + Fl

    def forward_2(self,x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv2(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv2(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv2(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width).to(x.device)).view(
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma2 * (out_H + out_W) + x

    def forward(self, Fl, Fv):
        out = self.forward_1( Fl, Fv)
        out = self.forward_2(out)
        out = self.forward_2(out)
        return out


if __name__ == '__main__':
    data = torch.randn(4,64,128,128)
    # net = RTB(in_channels=64)
    # net = CrissCrossRTB(in_channels=64)
    net = CrissCrossRTB_v2(in_channels=64)
    out = net(data,data) # (4, 128, 128, 128)
    print(out.shape)


