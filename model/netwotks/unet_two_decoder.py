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

class Decoder4Segformer(nn.Module):
    def __init__(self, params):
        super(Decoder4Segformer, self).__init__()
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

class Decoder4Segformer_add_Decoder(nn.Module):
    def __init__(self, params,backbone = 'mit'):
        super(Decoder4Segformer_add_Decoder, self).__init__()
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
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        other_up1 = F.interpolate(decoder_feature[0], size=x_up1.size()[2:], mode='bilinear')
        x_up1 += other_up1

        x_up2 = self.up2(x_up1, x2)
        other_up2 = F.interpolate(decoder_feature[1], size=x_up2.size()[2:], mode='bilinear')
        x_up2 += other_up2

        x_up3 = self.up3(x_up2, x1)
        other_up3 = F.interpolate(decoder_feature[2], size=x_up3.size()[2:], mode='bilinear')
        x_up3 += other_up3

        x_up4 = self.up4(x_up3)
        other_up4 = F.interpolate(decoder_feature[3], size=x_up4.size()[2:], mode='bilinear')
        x_up4 += other_up4


        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]


class Decoder4Segformer_cat_Decoder(nn.Module):
    def __init__(self, params,backbone = 'b2'
                                         ''):
        super(Decoder4Segformer_cat_Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_3input(self.ft_chns[4], self.ft_chns[3],self.ft_chns[3],self.ft_chns[3],
                                  dropout_p=0.0, mode_upsampling=self.up_type,backbone=backbone)
        self.up2 = UpBlock_3input(self.ft_chns[3], self.ft_chns[2],self.ft_chns[2],self.ft_chns[2],
                                  dropout_p=0.0, mode_upsampling=self.up_type,backbone=backbone)
        self.up3 = UpBlock_3input(self.ft_chns[2], self.ft_chns[1],self.ft_chns[1],self.ft_chns[1],
                                  dropout_p=0.0, mode_upsampling=self.up_type,backbone=backbone)

        if backbone in ['resnet50']:
            self.up4 = Double_UpBlock_3input(self.ft_chns[1] + self.ft_chns[0] * 2, self.ft_chns[0])
        else:
            self.up4 = Double_UpBlock_3input(self.ft_chns[1] * 2,self.ft_chns[0])


        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        other_up1 = F.interpolate(decoder_feature[0], size=x4.size()[2:], mode='bilinear')
        other_up2 = F.interpolate(decoder_feature[1], size=x3.size()[2:], mode='bilinear')
        other_up3 = F.interpolate(decoder_feature[2], size=x2.size()[2:], mode='bilinear')
        other_up4 = F.interpolate(decoder_feature[3], size=x1.size()[2:], mode='bilinear')



        x_up1 = self.up1(x4, x3 , other_up1)

        x_up2 = self.up2(x_up1, x2 , other_up2)

        x_up3 = self.up3(x_up2, x1 , other_up3)

        x_up4 = self.up4(x_up3 , other_up4,other_up4)


        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]

class Decoder4Segformer_rtb_Decoder(nn.Module):
    def __init__(self, params,rbt_layer = 4):
        super(Decoder4Segformer_rtb_Decoder, self).__init__()
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

        self.rbt_layer = rbt_layer
        self.rbt = RTB(self.ft_chns[4 - rbt_layer])

        self.out_conv = nn.Conv2d(self.ft_chns[0] , self.n_class, kernel_size=3, padding=1)
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        if self.rbt_layer == 1:
            other_up1 = F.interpolate(decoder_feature[0], size=x_up1.size()[2:], mode='bilinear')
            x_up1 = self.rbt(x_up1,other_up1)


        x_up2 = self.up2(x_up1, x2)
        if self.rbt_layer == 2:
            other_up2 = F.interpolate(decoder_feature[1], size=x_up2.size()[2:], mode='bilinear')
            x_up2 = self.rbt(x_up2,other_up2)


        x_up3 = self.up3(x_up2, x1)
        if self.rbt_layer == 3:
            other_up3 = F.interpolate(decoder_feature[2], size=x_up3.size()[2:], mode='bilinear')
            x_up3= self.rbt(x_up3,other_up3)

        x_up4 = self.up4(x_up3)
        if self.rbt_layer == 4:
            other_up4 = F.interpolate(decoder_feature[3], size=x_up4.size()[2:], mode='bilinear')
            x_up4 = self.rbt(x_up4,other_up4)

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]


class Decoder4Segformer_ccrtb_Decoder(nn.Module):
    def __init__(self, params,rbt_layer = 4):
        super(Decoder4Segformer_ccrtb_Decoder, self).__init__()
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

        self.rbt_layer = rbt_layer
        self.rbt = CrissCrossRTB(self.ft_chns[4 - rbt_layer])

        self.out_conv = nn.Conv2d(self.ft_chns[0] , self.n_class, kernel_size=3, padding=1)
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        if self.rbt_layer == 1:
            other_up1 = F.interpolate(decoder_feature[0], size=x_up1.size()[2:], mode='bilinear')
            x_up1 = self.rbt(x_up1,other_up1)


        x_up2 = self.up2(x_up1, x2)
        if self.rbt_layer == 2:
            other_up2 = F.interpolate(decoder_feature[1], size=x_up2.size()[2:], mode='bilinear')
            x_up2 = self.rbt(x_up2,other_up2)


        x_up3 = self.up3(x_up2, x1)
        if self.rbt_layer == 3:
            other_up3 = F.interpolate(decoder_feature[2], size=x_up3.size()[2:], mode='bilinear')
            x_up3= self.rbt(x_up3,other_up3)

        x_up4 = self.up4(x_up3)
        if self.rbt_layer == 4:
            other_up4 = F.interpolate(decoder_feature[3], size=x_up4.size()[2:], mode='bilinear')
            x_up4 = self.rbt(x_up4,other_up4)

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]


class Decoder4Segformer_ccrtbv2_Decoder(nn.Module):
    def __init__(self, params,rbt_layer = 4):
        super(Decoder4Segformer_ccrtbv2_Decoder, self).__init__()
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

        self.rbt_layer = rbt_layer
        self.rbt = CrissCrossRTB_v2(self.ft_chns[4 - rbt_layer])

        self.out_conv = nn.Conv2d(self.ft_chns[0] , self.n_class, kernel_size=3, padding=1)
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        if self.rbt_layer == 1:
            other_up1 = F.interpolate(decoder_feature[0], size=x_up1.size()[2:], mode='bilinear')
            x_up1 = self.rbt(x_up1,other_up1)


        x_up2 = self.up2(x_up1, x2)
        if self.rbt_layer == 2:
            other_up2 = F.interpolate(decoder_feature[1], size=x_up2.size()[2:], mode='bilinear')
            x_up2 = self.rbt(x_up2,other_up2)


        x_up3 = self.up3(x_up2, x1)
        if self.rbt_layer == 3:
            other_up3 = F.interpolate(decoder_feature[2], size=x_up3.size()[2:], mode='bilinear')
            x_up3= self.rbt(x_up3,other_up3)

        x_up4 = self.up4(x_up3)
        if self.rbt_layer == 4:
            other_up4 = F.interpolate(decoder_feature[3], size=x_up4.size()[2:], mode='bilinear')
            x_up4 = self.rbt(x_up4,other_up4)

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]


class Decoder4Segformer_ALLccrtb_Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder4Segformer_ALLccrtb_Decoder, self).__init__()
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

        self.rbt1 = CrissCrossRTB(self.ft_chns[4 - 1])
        self.rbt2 = CrissCrossRTB(self.ft_chns[4 - 2])
        self.rbt3 = CrissCrossRTB(self.ft_chns[4 - 3])
        self.rbt4 = CrissCrossRTB(self.ft_chns[4 - 4])

        self.out_conv = nn.Conv2d(self.ft_chns[0] , self.n_class, kernel_size=3, padding=1)
    def forward(self, feature,decoder_feature):
        x1 = feature[0]
        x2 = feature[1]
        x3 = feature[2]
        x4 = feature[3]

        x_up1 = self.up1(x4, x3)
        other_up1 = F.interpolate(decoder_feature[0], size=x_up1.size()[2:], mode='bilinear')
        x_up1 = self.rbt1(x_up1,other_up1)


        x_up2 = self.up2(x_up1, x2)
        other_up2 = F.interpolate(decoder_feature[1], size=x_up2.size()[2:], mode='bilinear')
        x_up2 = self.rbt2(x_up2,other_up2)


        x_up3 = self.up3(x_up2, x1)
        other_up3 = F.interpolate(decoder_feature[2], size=x_up3.size()[2:], mode='bilinear')
        x_up3= self.rbt3(x_up3,other_up3)

        x_up4 = self.up4(x_up3)
        other_up4 = F.interpolate(decoder_feature[3], size=x_up4.size()[2:], mode='bilinear')
        x_up4 = self.rbt4(x_up4,other_up4)

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]



class Decoder_add_Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder_add_Decoder, self).__init__()
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

    def forward(self, feature,decoder_feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x_up1 = self.up1(x4, x3)
        other_up1 = F.interpolate(decoder_feature[0],size=x_up1.size()[2:],mode='bilinear')
        x_up1 += other_up1

        x_up2 = self.up2(x_up1, x2)
        other_up2 = F.interpolate(decoder_feature[1],size=x_up2.size()[2:],mode='bilinear')
        x_up2 += other_up2

        x_up3 = self.up3(x_up2, x1)
        other_up3 = F.interpolate(decoder_feature[2],size=x_up3.size()[2:],mode='bilinear')
        x_up3 += other_up3

        x_up4 = self.up4(x_up3, x0)
        other_up4 = F.interpolate(decoder_feature[3],size=x_up4.size()[2:],mode='bilinear')
        x_up4 += other_up4

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]

class Decoder_cat_Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder_cat_Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_3input(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type,backbone='org')
        self.up2 = UpBlock_3input(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2],self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type,backbone='org')
        self.up3 = UpBlock_3input(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1],self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type,backbone='org')
        self.up4 = UpBlock_3input(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0],self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type,backbone='org')
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature,decoder_feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        other_up1 = F.interpolate(decoder_feature[0], size=x3.size()[2:], mode='bilinear')
        other_up2 = F.interpolate(decoder_feature[1], size=x2.size()[2:], mode='bilinear')
        other_up3 = F.interpolate(decoder_feature[2], size=x1.size()[2:], mode='bilinear')
        other_up4 = F.interpolate(decoder_feature[3], size=x0.size()[2:], mode='bilinear')

        x_up1 = self.up1(x4, x3,other_up1)

        x_up2 = self.up2(x_up1, x2,other_up2)

        x_up3 = self.up3(x_up2, x1,other_up3)

        x_up4 = self.up4(x_up3, x0,other_up4)

        output = self.out_conv(x_up4)
        return [x_up1,x_up2,x_up3,x_up4,output]


class UNet_MiT(nn.Module):
    def __init__(self, in_chns, class_num,phi='b2',pretrained=True):
        super(UNet_MiT, self).__init__()

        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.encoder = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)


        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 64, 128, 320, 512],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.decoder1 = Decoder4Segformer(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1[-1]


class UNet_ResNet(nn.Module):
    def __init__(self, in_chns, class_num,phi='resnet50',pretrained=True):
        super(UNet_ResNet, self).__init__()

        self.in_channels = {
            'resnet18': [64, 128, 256, 512], 'resnet34': [64, 128,256,512], 'resnet50': [256,512,1024,2048],
            'resnet101': [256,512,1024,2048],
        }[phi]
        self.encoder = {
            'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
            'resnet101': resnet101,
        }[phi](pretrained)


        params1 = {'in_chns': in_chns,
                  # 'feature_chns': [16].extend(self.in_channels),
                  'feature_chns': [16,256,512,1024,2048],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.decoder1 = Decoder4Segformer(params1)

    def forward(self, x):
        feature = self.encoder.base_forward(x)
        output1 = self.decoder1(feature)
        return output1[-1]



# class UNet_two_Decoder(nn.Module):
#     def __init__(self, in_chns, class_num1,class_num2,fuse_type=None):
#         super(UNet_two_Decoder, self).__init__()
#
#         params1 = {'in_chns': in_chns,
#                   'feature_chns': [16, 32, 64, 128, 256],
#                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                   'class_num': class_num2,
#                   'up_type': 1,
#                   'acti_func': 'relu'}
#         params2 = {'in_chns': in_chns,
#                   'feature_chns': [16, 32, 64, 128, 256],
#                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
#                   'class_num': class_num1,
#                   'up_type': 1,
#                   'acti_func': 'relu'}
#
#         self.fuse_type = fuse_type
#         self.encoder = Encoder(params1)
#         self.decoder1 = Decoder(params1)
#         if fuse_type == 'add':
#             self.decoder2 = Decoder_add_Decoder(params2)
#         elif fuse_type == 'cat':
#             self.decoder2 = Decoder_add_Decoder(params2)
#         else:
#             self.decoder2 = Decoder(params2)
#
#     def forward(self, x):
#         feature = self.encoder(x)
#         output_decoder1 = self.decoder1(feature)
#         if self.fuse_type is not None:
#             output_decoder2 = self.decoder2(feature,output_decoder1)
#         else:
#             output_decoder2 = self.decoder2(feature)
#         return output_decoder2[-1],output_decoder1[-1]


class UNet_MiT_two_Decoder(nn.Module):
    def __init__(self, in_chns, class_num1,class_num2,phi='b2',fuse_type=None,pretrained=True):
        super(UNet_MiT_two_Decoder, self).__init__()

        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.encoder = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]


        params1 = {'in_chns': in_chns,
                  'feature_chns': [32, 64, 128, 320, 512],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num2,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [32, 64, 128, 320, 512],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num1,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.fuse_type = fuse_type

        self.decoder1 = Decoder4Segformer(params1)
        if fuse_type == 'add':
            self.decoder2 = Decoder4Segformer_add_Decoder(params2)
        elif fuse_type == 'cat':
            self.decoder2 = Decoder_add_Decoder(params2)
        else:
            self.decoder2 = Decoder(params2)

    def forward(self, x):
        feature = self.encoder.forward(x)
        output_decoder1 = self.decoder1(feature)
        if self.fuse_type is not None:
            output_decoder2 = self.decoder2(feature,output_decoder1)
        else:
            output_decoder2 = self.decoder2(feature)
        return output_decoder2[-1],output_decoder1[-1]


class UNet_two_Decoder(nn.Module):
    def __init__(self, in_chns, class_num1,class_num2,phi='b2',fuse_type=None,pretrained=True):
        super(UNet_two_Decoder, self).__init__()

        self.phi = phi
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
            'resnet18': [64, 128, 256, 512], 'resnet34': [64, 128, 256, 512],
            'resnet50': [256, 512, 1024, 2048],'resnet101': [256, 512, 1024, 2048],
            'org': [16, 32, 64, 128, 256],
        }[phi]

        self.encoder = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
            'resnet18': resnet18, 'resnet34': resnet34,
            'resnet50': resnet50, 'resnet101': resnet101,
            'org': resnet50,
        }[phi](pretrained)

        if phi == 'org':
            feature_chns = self.in_channels
        else:
            feature_chns = [32]
            feature_chns.extend(self.in_channels)

        params1 = {'in_chns': in_chns,
                  'feature_chns': feature_chns,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num2,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': feature_chns,
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num1,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.fuse_type = fuse_type

        if phi == 'org':
            self.encoder = Encoder(params1)
            self.decoder1 = Decoder(params1)
            if fuse_type == 'add':
                self.decoder2 = Decoder_add_Decoder(params2)
            elif fuse_type == 'cat':
                self.decoder2 = Decoder_cat_Decoder(params2)
            else:
                self.decoder2 = Decoder(params2)
        else:
            self.decoder1 = Decoder4Segformer(params1)
            if fuse_type == 'add':
                self.decoder2 = Decoder4Segformer_add_Decoder(params2)
            elif fuse_type == 'cat':
                self.decoder2 = Decoder4Segformer_cat_Decoder(params2,backbone = phi)
            elif 'rtb' in fuse_type:
                rbt_layer = int(fuse_type.split('rtb')[-1])
                if 'ccrtb' in fuse_type:
                    if 'allccrtb' in fuse_type:
                        self.decoder2 = Decoder4Segformer_ALLccrtb_Decoder(params2)
                    elif 'v2_ccrtb' in fuse_type:
                        self.decoder2 = Decoder4Segformer_ccrtbv2_Decoder(params2, rbt_layer=rbt_layer)
                    else:
                        self.decoder2 = Decoder4Segformer_ccrtb_Decoder(params2, rbt_layer=rbt_layer)
                else:
                    self.decoder2 = Decoder4Segformer_rtb_Decoder(params2,rbt_layer=rbt_layer)
            else:
                self.decoder2 = Decoder4Segformer(params2)

    def forward(self, x):
        if 'resnet' in self.phi:
            feature = self.encoder.base_forward(x)
        else:
            feature = self.encoder.forward(x)
        output_decoder1 = self.decoder1(feature)
        if self.fuse_type in ['add','cat',
                              'rtb1','rtb2','rtb3','rtb4',
                              'ccrtb1','ccrtb2','ccrtb3','ccrtb4',
                              'v2_ccrtb1','v2_ccrtb2','v2_ccrtb3','v2_ccrtb4',
                              'allccrtb0'] :
            output_decoder2 = self.decoder2(feature,output_decoder1)
        else:
            output_decoder2 = self.decoder2(feature)
        return output_decoder2[-1],output_decoder1[-1]



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

    model = UNet_two_Decoder(in_chns=3, class_num1=3,class_num2=2,phi='b2',pretrained=False,fuse_type='v2_ccrtb1')
    out,_ = model(input_data)
    print(out.shape)
