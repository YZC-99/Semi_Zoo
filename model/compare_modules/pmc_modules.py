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
        self.dab_block = DABBlock(in_channels)

        # Final convolution
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
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

if __name__ == '__main__':
    # Example usage DAB
    # in_channels = 64  # Adjust based on your input channels
    # dab_block = DABBlock(in_channels)
    # input_tensor = torch.randn(1, in_channels, 32, 32)  # Adjust based on your input size
    # output_tensor = dab_block(input_tensor)
    # print(output_tensor.shape)

    # Example usage
    in_channels = 64  # Adjust based on your input channels
    pff_3 = PFF3(in_channels)
    pre_tensor = torch.randn(1, in_channels, 64, 64)  # Adjust based on your input size
    cur_tensor = torch.randn(1, in_channels, 32, 32)
    nex_tensor = torch.randn(1, in_channels, 16, 16)
    output_tensor = pff_3(pre_tensor, cur_tensor, nex_tensor)
    print(output_tensor.shape)