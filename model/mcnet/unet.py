# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

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
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
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

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

class Decoder_v2(nn.Module):
    def __init__(self, params):
        super(Decoder_v2, self).__init__()
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

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        return output,x

class Decoder_dtc(nn.Module):
    def __init__(self, params,outchannel_minus1 = False):
        super(Decoder_dtc, self).__init__()
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

        out_channel = self.n_class
        if outchannel_minus1:
            out_channel = self.n_class - 1
        self.out_conv = nn.Conv2d(self.ft_chns[0], out_channel, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(self.ft_chns[0], out_channel, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        output2 = self.out_conv2(x)
        out_tanh = self.tanh(output2)
        return output,out_tanh

class Decoder_dtc_gmm(nn.Module):
    def __init__(self, params,outchannel_minus1 = False):
        super(Decoder_dtc_gmm, self).__init__()
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

        out_channel = self.n_class
        if outchannel_minus1:
            out_channel = self.n_class - 1
        self.out_conv = nn.Conv2d(self.ft_chns[0], out_channel, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(self.ft_chns[0], out_channel, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        output2 = self.out_conv2(x)
        out_tanh = self.tanh(output2)
        return output,out_tanh,x
    
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1


class UNet_two_Decoder(nn.Module):
    def __init__(self, in_chns, class_num1,class_num2):
        super(UNet_two_Decoder, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num1,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num2,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder1(feature)
        return output1,output2


class UNet_DTC2d(nn.Module):
    def __init__(self, in_chns, class_num,outchannel_minus1 = False):
        super(UNet_DTC2d, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder_dtc(params1,outchannel_minus1 = False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        feature = self.encoder(x)
        output1,output_tanh_1 = self.decoder1(feature)
        return {"output1":output1,
                "output_tanh_1":output_tanh_1}


class UNet_DTC_GMM2d(nn.Module):
    def __init__(self, in_chns, class_num,outchannel_minus1 = False):
        super(UNet_DTC_GMM2d, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder_dtc_gmm(params1,outchannel_minus1 = outchannel_minus1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        feature = self.encoder(x)
        output1,output_tanh_1,feat = self.decoder1(feature)
        return {"output1":output1,
                "output_tanh_1":output_tanh_1,
                "feat":feat,
                }


class MCNet2d_compete_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_compete_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output_tanh1 = self.tanh1(output1)
        output_tanh2 = self.tanh2(output2)
        return {
            "output1":output1,
            "output2":output2,
        }


class MCNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3

class MCNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 2,
                  'acti_func': 'relu'}
        params4 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 3,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.decoder3 = Decoder(params3)
        self.decoder4 = Decoder(params4)
        
    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)    
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4




class MCNet2d_compete_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_compete_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output_tanh1 = self.tanh1(output1)
        output_tanh2 = self.tanh2(output2)
        return {
            "output1":output1,
            "output2":output2,
        }


class MCNet2d_tanh_v1(nn.Module):
    def __init__(self, in_chns, class_num):
        super(MCNet2d_tanh_v1, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 0,
                   'acti_func': 'relu'}
        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder(params2)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output_tanh1 = self.tanh1(output1)
        output_tanh2 = self.tanh2(output2)
        return {
            "output1":output1,
            "output2":output2,
            "output_tanh1":output_tanh1,
            "output_tanh2":output_tanh2,
        }


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    # from ptflops import get_model_complexity_info
    # model = UNet(in_chns=3, class_num=3)
    # model = MCNet2d_tanh_v1(in_chns=3, class_num=3)
    model = UNet(in_chns=3, class_num=3)
    # with torch.cuda.device(0):
    #   macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
    #                                            print_per_layer_stat=True, verbose=True)
    #   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #   print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # import ipdb; ipdb.set_trace()
    input_data = torch.randn(2,3,256,256,dtype=torch.float32)
    out = model(input_data)
    print(out.keys())
