import torch
from torch import nn
from torch.nn import functional as F


class MSFE(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(in_channels, in_channels, 1, bias=False))
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                     nn.Conv2d(in_channels, in_channels, 3,1,1, bias=False),
                                     ))
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False),
                                     nn.Conv2d(in_channels, in_channels, 5,1,2, bias=False),
                                     ))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Conv2d(3*in_channels,in_channels,1, bias=False)
        # self.up = nn.Upsample(scale_factor)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        # out = self.up(res)
        return res
