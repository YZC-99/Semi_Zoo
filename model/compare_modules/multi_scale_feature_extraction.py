import torch
from torch import nn
from torch.nn import functional as F


class MSFE(nn.Sequential):
    def __init__(self, in_channels,scale_factor):
        modules = []
        modules.append(nn.Conv2d(in_channels, in_channels, 1, bias=False))
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                     nn.Conv2d(in_channels, in_channels, 3,1,1, bias=False),
                                     ))
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                     nn.Conv2d(in_channels, in_channels, 5,1,2, bias=False),
                                     ))
        self.convs = nn.ModuleList(modules)

        self.up = nn.Upsample(scale_factor)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.up(res)
