# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function
from model.backbone.mit import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from model.backbone.resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from model.compare_modules.rtb import RTB,CrissCrossRTB,CrissCrossRTB_v2
import torch.nn.functional as F
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ODOC_inter_Vessel(nn.Module):
    def __init__(self, in_channels, out_channels, reduction = 16):
        super(ODOC_inter_Vessel, self).__init__()
        self.se1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)
        )
        self.se2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, odoc_features,vessel_features):
        # 需要保证odoc_features和vessel_features的形状和通道一模一样
        out_multi = odoc_features * vessel_features
        out_add = odoc_features + vessel_features

        #
        out_se1 = self.se1(torch.cat([out_multi,out_add,odoc_features],dim=1))

        # 需要out_se1和odoc_features的形状和通道一模一样
        out_se1_add = out_se1 + odoc_features

        out_se2 = self.se2(out_se1_add)

        # 需要out_se2和odoc_features的通道数一模一样
        out = out_se2 + odoc_features

        return out




class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        if x1.size()[-2:] != x2.size()[-2:]:
            x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UpBlock_3input(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2,in_channels3, out_channels, dropout_p, mode_upsampling=1,backbone='b2'):
        super(UpBlock_3input, self).__init__()
        self.mode_upsampling = mode_upsampling
        self.backbone = backbone
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 3, out_channels, dropout_p)

    def forward(self, x1,x2,x3):
        if self.backbone in ['b2','b4']:
            if self.mode_upsampling != 0:
                x1 = self.conv1x1(x1)
            x1 = self.up(x1)
            x3 = self.up(x3)
            x = torch.cat([x3,x2, x1], dim=1)
            x = self.conv(x)
        elif self.backbone in ['resnet50','resnet34']:
            if self.mode_upsampling != 0:
                x1 = self.conv1x1(x1)
            if x1.size()[-2:] != x2.size()[-2:]:
                x1 = self.up(x1)
            if x3.size()[-2:] != x2.size()[-2:]:
                x3 = self.up(x3)
            x = torch.cat([x3,x2, x1], dim=1)
            x = self.conv(x)
        else:
            if self.mode_upsampling != 0:
                x1 = self.conv1x1(x1)
            if x1.size()[-2:] != x2.size()[-2:]:
                x1 = self.up(x1)
            x = torch.cat([x3,x2, x1], dim=1)
            x = self.conv(x)
        return x

class Double_UpBlock_3input(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels,out_channels):
        super(Double_UpBlock_3input, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
    def forward(self, x1,x2,x3):
        x = torch.cat([x3,x2, x1], dim=1)
        x = self.up(x)
        return x



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3, x0)
        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]

class Decoder4Dual_encoder(nn.Module):
    def __init__(self, params):
        super(Decoder4Dual_encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = nn.Sequential(
            nn.Conv2d(self.ft_chns[1], self.ft_chns[1], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(self.ft_chns[1]),
            nn.Conv2d(self.ft_chns[1], self.ft_chns[0], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
    def forward(self, feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        x_up2 = self.up2(x_up1, x2)
        x_up3 = self.up3(x_up2, x1)
        x_up4 = self.up4(x_up3)
        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]





class UNet_Dual_Encoder(nn.Module):
    def __init__(self, in_chns, class_num,phi='resnet50',phi2 = 'resnet34',pretrained=True):
        super(UNet_Dual_Encoder, self).__init__()

        self.in_channels = {
            'resnet18': [64, 128, 256, 512], 'resnet34': [64, 128,256,512], 'resnet50': [256,512,1024,2048],
            'resnet101': [256,512,1024,2048],
        }[phi]
        self.encoder = {
            'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
            'resnet101': resnet101,
        }[phi](pretrained)

        self.encoder2 = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi2](pretrained)

        # feature_chns = [16]
        # feature_chns.extend(self.in_channels)
        feature_chns = [16, 64, 128, 256, 1024]
        params1 = {'in_chns': in_chns,
                  # 'feature_chns': [16].extend(self.in_channels),
                  'feature_chns': feature_chns,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.decoder1 = Decoder4Dual_encoder(params1)

    def forward(self, x):
        feature = self.encoder.base_forward(x)
        feature2 = self.encoder2(x)
        feature[-1] = torch.cat([feature[-1],feature2[-1]],dim=1)
        output1 = self.decoder1(feature)
        return output1[-1]




if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    # from ptflops import get_model_complexity_info
    # model = UNet(in_chns=3, class_num=3)
    # model = MCNet2d_tanh_v1(in_chns=3, class_num=3)
    # model = UNet(in_chns=3, class_num=3)
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import ipdb; ipdb.set_trace()
    input_data = torch.randn(2,3,256,256,dtype=torch.float32)
    # model = UNet_MiT(in_chns=3, class_num=3,pretrained=False)
    # model = UNet_ResNet(in_chns=3, class_num=3,pretrained=False)
    # out = model(input_data)

    model = UNet_Dual_Encoder(in_chns=3, class_num=3,phi='resnet34',phi2='b2',pretrained=False)
    out,_ = model(input_data)
    print(out.shape)
