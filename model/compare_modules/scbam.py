import torch
import torch.nn as nn
from torch.nn import Softmax
from torch.nn import init
from model.module.attentions import SCCBAMMModule


class SCCBAM_Module(nn.Module):

    def __init__(self, in_channels_list):
        super(SCCBAM_Module, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(in_channels_list,1):
            sa_layer = 'ska_layer{}'.format(idx)
            sa_layer_module = SCCBAMMModule( in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)

    def forward(self, fpn_features):
        for idx,(feature,sa) in enumerate(zip(fpn_features,self.sa_layers)):
            feature_out = getattr(self,sa)(feature)
            fpn_features[idx] = feature_out
        return fpn_features