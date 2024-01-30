import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

class Mamba_block(nn.Module):

    def __init__(self, in_channels):
        super(Mamba_block, self).__init__()
        from mamba_ssm import Mamba
        self.ln = nn.LayerNorm(in_channels)
        self.mamba = Mamba(
            d_model=in_channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, features):
        middle_feature = features
        B, C = middle_feature.shape[:2]
        n_tokens = middle_feature.shape[2:].numel()
        img_dims = middle_feature.shape[2:]
        middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2)
        middle_feature_flat = self.ln(middle_feature_flat)
        out = self.mamba(middle_feature_flat)
        out = out.transpose(-1, -2).view(B, C, *img_dims)
        return out

class Mamba_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(Mamba_Module, self).__init__()
        from mamba_ssm import Mamba

        self.mamba_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            mamba_layer = 'mamba_layer{}'.format(idx)
            mamba_layer_module = Mamba_block(in_channels)
            self.add_module(mamba_layer,mamba_layer_module)
            self.mamba_layers.append(mamba_layer)

    def forward(self, features):
        for idx,(feature,mamba) in enumerate(zip(features,self.mamba_layers)):
            feature_out = getattr(self,mamba)(feature)
            features[idx] = feature_out
        return features