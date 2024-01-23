import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet

class DABBlock(nn.Module):
    def __init__(self, in_channels):
        super(DABBlock, self).__init__()

        # CA
        self.ca_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ca_dense = nn.Linear(in_channels, in_channels)
        self.ca_sigmoid = nn.Sigmoid()

        # SA
        self.sa_avg = nn.AdaptiveAvgPool2d(1)
        self.sa_sigmoid = nn.Sigmoid()

        # PA
        self.pa_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.pa_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # CA
        ca_avg = self.ca_avgpool(x).view(x.size(0), -1)
        ca_score = self.ca_sigmoid(self.ca_dense(ca_avg)).view(x.size(0), -1, 1, 1)
        CA = x * ca_score

        # SA
        sa_avg = self.sa_avg(x).view(x.size(0), -1, 1, 1)
        sa_score = self.sa_sigmoid(sa_avg)
        SA = x * sa_score

        # PA
        pa_score = self.pa_sigmoid(self.pa_conv(x))
        PA = x * pa_score

        # Output
        out = CA + SA + PA
        return out

class DABModules(nn.Module):
    def __init__(self, in_channels_list):
        super(DABModules, self).__init__()
        self.dab_blocks = []
        for idx,in_channels in enumerate(in_channels_list,1):
            dab_block = 'DAB_Block{}'.format(idx)
            dab_module = DABBlock(in_channels)
            self.add_module(dab_block,dab_module)
            self.dab_blocks.append(dab_block)


    def forward(self, features):
        for idx,dab in enumerate(self.dab_blocks):
            features[idx] = getattr(self,dab)(features[idx])
        return features



class PFF3(nn.Module):
    def __init__(self, in_channels):
        super(PFF3, self).__init__()

        # Down
        self.down_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.down_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Up
        self.up_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.up_conv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Attention
        self.dab_block = DABBlock(in_channels * 2)

        # Final convolution
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, pre, cur, nex):
        # Down
        down = self.down_conv1(pre)
        down = self.down_conv2(down)

        # Up
        up = self.up_conv1(nex)
        up = self.up_conv2(up)

        # Element-wise multiplication
        mul_low = down * cur
        mul_high = up * cur

        # Concatenate
        concat = torch.cat([mul_low, mul_high], dim=1)

        # Attention
        att = self.dab_block(concat)

        # Final convolution
        x = self.final_conv(att)
        x = self.leaky_relu(x)

        return x



class PFF2Pre(nn.Module):
    def __init__(self, in_channels):
        super(PFF2Pre, self).__init__()

        # Up
        self.up_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.up_conv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Attention
        self.dab_block = DABBlock(in_channels * 2)

        # Final convolution
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, cur, nex):
        # Up
        up = self.up_conv1(nex)
        up = self.up_conv2(up)

        # Element-wise multiplication
        mul = torch.mul(cur, up)

        # Concatenate along the channel dimension (dim=1)
        concat = torch.cat([cur, mul], dim=1)

        # Attention
        att = self.dab_block(concat)

        # Final convolution
        x = self.final_conv(att)
        x = self.leaky_relu(x)

        return x


class PFF2Nex(nn.Module):
    def __init__(self, in_channels):
        super(PFF2Nex, self).__init__()

        # Down
        self.down_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.down_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Attention
        self.dab_block = DABBlock(in_channels * 2)

        # Final convolution
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, pre, cur):
        # Down
        down = self.down_conv1(pre)
        down = self.down_conv2(down)

        # Element-wise multiplication
        mul = torch.mul(down, cur)

        # Concatenate along the channel dimension (dim=1)
        concat = torch.cat([cur, mul], dim=1)

        # Attention
        att = self.dab_block(concat)

        # Final convolution
        x = self.final_conv(att)
        x = self.leaky_relu(x)
        return x




class PMC_Modules(nn.Module):
    def __init__(self, in_channels_list):
        super(PMC_Modules, self).__init__()

        self.PFF_first = PFF2Pre(in_channels_list[0])
        self.PFF_last = PFF2Nex(in_channels_list[-1])

        self.pff_layers = []

        for idx,in_channels in enumerate(in_channels_list[1:-1],1):
            pff_layer = 'pff_layer{}'.format(idx)
            pff_layer_module = PFF3(in_channels)
            self.add_module(pff_layer,pff_layer_module)
            self.pff_layers.append(pff_layer)



    def forward(self, features):
        features[0] = self.PFF_first(features[0],features[1])
        features[-1] = self.PFF_last(features[-2],features[-1])
        for idx,pff in enumerate(self.pff_layers,1):
            features[idx] = pff(features[idx - 1],features[idx],features[idx + 1])

        return features


if __name__ == '__main__':
    # Example usage DAB
    # in_channels = 64  # Adjust based on your input channels
    # dab_block = DABBlock(in_channels)
    # input_tensor = torch.randn(1, in_channels, 32, 32)  # Adjust based on your input size
    # output_tensor = dab_block(input_tensor)
    # print(output_tensor.shape)

    # Example usage PFF3
    # in_channels = 64  # Adjust based on your input channels
    # pff_3 = PFF3(in_channels)
    # pre_tensor = torch.randn(4, in_channels, 64, 64)  # Adjust based on your input size
    # cur_tensor = torch.randn(4, in_channels, 32, 32)
    # nex_tensor = torch.randn(4, in_channels, 16, 16)
    # output_tensor = pff_3(pre_tensor, cur_tensor, nex_tensor)
    # print(output_tensor.shape)
    #

    # # Example usage Pre
    # in_channels = 64  # Adjust based on your input channels
    # pff_2_pre = PFF2Pre(in_channels)
    # cur_tensor = torch.randn(1, in_channels, 64, 64)  # Adjust based on your input size
    # nex_tensor = torch.randn(1, in_channels, 32, 32)
    # output_tensor = pff_2_pre(cur_tensor, nex_tensor)
    # print(output_tensor.shape)

    # Example usage
    # in_channels = 64  # Adjust based on your input channels
    # pff_2_nex = PFF2Nex(in_channels)
    # pre_tensor = torch.randn(1, in_channels, 64, 64)  # Adjust based on your input size
    # cur_tensor = torch.randn(1, in_channels, 32, 32)
    # output_tensor = pff_2_nex(pre_tensor, cur_tensor)
    # print(output_tensor.shape)


    # Example usage
    in_channels_list = [32,64,128,256]  # Adjust based on your input channels
    pff_2_nex = DABModules(in_channels_list)
    input_tensor = [torch.randn(1,32,512,512),
                    torch.randn(1,64,256,256),
                    torch.randn(1, 128, 128, 128),
                    torch.randn(1, 256, 64, 64)
                    ]
    output_tensor = pff_2_nex(input_tensor)
    print(output_tensor[0].shape)
