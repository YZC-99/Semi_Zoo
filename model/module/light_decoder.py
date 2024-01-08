import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


        self.conv = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )


    def forward(self, x, skip=None):
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.up(x)

        if skip is not None:
            x = x + skip
        x = self.conv(x)

        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class Light_Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        # if n_blocks != len(decoder_channels):
        #     raise ValueError(
        #         "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
        #             n_blocks, len(decoder_channels)
        #         )
        #     )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[-4:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        self.convc5 = nn.Sequential(
            md.Conv2dReLU(256,128,3,1,1,use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
            md.Conv2dReLU(128, 128, 3, 1, 1, use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
            md.Conv2dReLU(128, 128, 3, 1, 1, use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.convc4 = nn.Sequential(
            md.Conv2dReLU(256,128,3,1,1,use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
            md.Conv2dReLU(128, 128, 3, 1, 1, use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.convc3 = nn.Sequential(
            md.Conv2dReLU(256,128,3,1,1,use_batchnorm=use_batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.convc2 = nn.Sequential(
            md.Conv2dReLU(256,128,3,1,1,use_batchnorm=use_batchnorm),
        )

    def forward(self, *features):

        out_c5 = self.convc5(features[-1])
        out_c4 = self.convc4(features[-2])
        out_c3 = self.convc3(features[-3])
        out_c2 = self.convc2(features[-4])

        out_feat = (out_c5 + out_c4 + out_c3 + out_c2) / 4.


        return out_feat
