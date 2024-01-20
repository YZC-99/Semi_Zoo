import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from model.module.attentions import CrissCrossAttention

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class RTFM(nn.Module):

    def __init__(self, d_model, d_k, d_v, h=12,dropout=.1):
        super(RTFM, self).__init__()
        self.sa = ScaledDotProductAttention( d_model, d_k, d_v, h,dropout)
        self.LN = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 3072),
            nn.Linear(3072, d_model),
            nn.ReLU()
        )

    def forward(self, queries, keys, values):
        sa_out = self.sa(queries, keys, values)

        residual_1 = queries + sa_out

        residual_1_out = self.LN(residual_1)

        mlp_out = self.mlp(residual_1_out)

        return mlp_out + residual_1


class SA_after_FPN(nn.Module):

    def __init__(self, fpn_out_channels, h=1,dropout=.1):
        super(SA_after_FPN, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(fpn_out_channels,1):
            sa_layer = 'sa_layer{}'.format(idx)
            sa_layer_module = ScaledDotProductAttention( in_channels, 64, 64, h,dropout)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)


    def forward(self, fpn_features1,fpn_features2):

        for idx,(feature1,feature2,sa) in enumerate(zip(fpn_features1,fpn_features2,self.sa_layers)):
            b,c,h,w = feature1.size()
            seq_feature1 = feature1.reshape(b, c, -1)
            seq_feature1 = seq_feature1.permute(0, 2, 1)
            seq_feature2 = feature2.reshape(b, c, -1)
            seq_feature2 = seq_feature2.permute(0, 2, 1)
            seq_output = getattr(self,sa)(seq_feature1,seq_feature1,seq_feature2)

            seq_output = seq_output.permute(0, 2, 1)
            sa_feature_output = seq_output.reshape(b, c, h, w)
            fpn_features1[idx] = sa_feature_output
        return fpn_features1



class SA_Criss_after_FPN(nn.Module):

    def __init__(self, fpn_out_channels, h=12,dropout=.1):
        super(SA_Criss_after_FPN, self).__init__()

        self.sa_layers = []
        for idx,in_channels in enumerate(fpn_out_channels,1):
            sa_layer = 'sa_layer{}'.format(idx)
            sa_layer_module = CrissCrossAttention(in_channels)
            self.add_module(sa_layer,sa_layer_module)
            self.sa_layers.append(sa_layer)


    def forward(self, fpn_features1,fpn_features2):
        new_features = []
        for idx,(feature1,feature2,sa) in enumerate(zip(fpn_features1,fpn_features2,self.sa_layers)):
            output = getattr(self,sa)(feature1,feature2)
            new_features.append(output)
        return tuple(new_features)




if __name__ == '__main__':
    odoc_feature = torch.randn(8,512,7,7)
    vessel_feature = torch.randn(8,512,7,7)
    b,c,h,w = odoc_feature.size()

    odoc_seq_input = odoc_feature.reshape(b,c,-1)
    odoc_seq_input = odoc_seq_input.permute(0,2,1)

    vessel_seq_input = vessel_feature.reshape(b,c,-1)
    vessel_seq_input = vessel_seq_input.permute(0,2,1)

    # seq_input=torch.randn(8,49,512)
    # sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    sa = RTFM(d_model=512, d_k=64, d_v=64)
    output=sa(odoc_seq_input,odoc_seq_input,vessel_seq_input)

    seq_output = output.permute(0, 2, 1)
    feature_out = seq_output.reshape(b,c,h,w)

    print(output.shape)
    print(feature_out.shape)
