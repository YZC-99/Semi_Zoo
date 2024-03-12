from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from model.compare_modules.rtfm import RTFM,ScaledDotProductAttention
from model.module.light_decoder import Light_Decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from model.module.gap import GlobalAvgPool2D
from model.module import fpn
from model.module.farseg import SceneRelation

class Unet_wFPN_wSR(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        if fpn_pretrained :
            # Update SceneRelation weights
            ckpt_apth = 'pretrained/farseg50.pth'
            sd = torch.load(ckpt_apth)
            fpn_state_dict = self.fpn.state_dict()
            for name, param in sd['model'].items():

                if 'module.fpn' in name:
                    # 移除 'module.' 前缀
                    name = name.replace('module.', '')
                    # Update SceneRelation state_dict
                    fpn_state_dict[name] = param
            # Load the modified SceneRelation state_dict

            self.fpn.load_state_dict(fpn_state_dict,strict=False)
            print("================加载FPN权重成功！===============")
            print(fpn_state_dict.keys())
            # --------------


        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.gap = GlobalAvgPool2D()
        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = UnetDecoder(
            encoder_channels=new_encoder_channels,
            decoder_channels= decoder_channels ,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])

        features[-4:] = self.fpn(features[-4:])

        features[-4:] = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


class Unet_wFPN_wPyramidMHSA_SR(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        if fpn_pretrained :
            # Update SceneRelation weights
            ckpt_apth = 'pretrained/farseg50.pth'
            sd = torch.load(ckpt_apth)
            fpn_state_dict = self.fpn.state_dict()
            for name, param in sd['model'].items():

                if 'module.fpn' in name:
                    # 移除 'module.' 前缀
                    name = name.replace('module.', '')
                    # Update SceneRelation state_dict
                    fpn_state_dict[name] = param
            # Load the modified SceneRelation state_dict

            self.fpn.load_state_dict(fpn_state_dict,strict=False)
            print("================加载FPN权重成功！===============")
            print(fpn_state_dict.keys())
            # --------------


        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.MHSA = ScaledDotProductAttention(sum(self.encoder.out_channels[-4:]), 128, 128, 12)
        self.gap = GlobalAvgPool2D()


        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            # in_channels=self.encoder.out_channels[-1],
                            in_channels=sum(self.encoder.out_channels[-4:]),
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = UnetDecoder(
            encoder_channels=new_encoder_channels,
            decoder_channels= decoder_channels ,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)


        gap_features = torch.cat([features[-4],
                                  F.interpolate(features[-3],size=features[-4].size()[-2:]),
                                  F.interpolate(features[-2],size=features[-4].size()[-2:]),
                                  F.interpolate(features[-1],size=features[-4].size()[-2:])],
                                 dim=1
                                 )



        b, c, h, w = gap_features.size()
        seq_deep_features = gap_features.reshape(b, c, -1)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        seq_deep_features = self.MHSA(seq_deep_features, seq_deep_features, seq_deep_features)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        gap_features = seq_deep_features.reshape(b, c, h, w)

        c_last = self.gap(gap_features)

        features[-4:] = self.fpn(features[-4:])

        features[-4:] = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks

class Unet_wFPN_wPyramidMHSA_SR_wLightDecoder(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        if fpn_pretrained :
            # Update SceneRelation weights
            ckpt_apth = 'pretrained/farseg50.pth'
            sd = torch.load(ckpt_apth)
            fpn_state_dict = self.fpn.state_dict()
            for name, param in sd['model'].items():

                if 'module.fpn' in name:
                    # 移除 'module.' 前缀
                    name = name.replace('module.', '')
                    # Update SceneRelation state_dict
                    fpn_state_dict[name] = param
            # Load the modified SceneRelation state_dict

            self.fpn.load_state_dict(fpn_state_dict,strict=False)
            print("================加载FPN权重成功！===============")
            print(fpn_state_dict.keys())
            # --------------


        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.MHSA = ScaledDotProductAttention(sum(self.encoder.out_channels[-4:]), 128, 128, 12)
        self.gap = GlobalAvgPool2D()


        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            # in_channels=self.encoder.out_channels[-1],
                            in_channels=sum(self.encoder.out_channels[-4:]),
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = Light_Decoder(
            encoder_channels=new_encoder_channels,
            decoder_channels= self.fpn_out_channels ,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=int(self.fpn_out_channels / 2),
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=4
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)


        gap_features = torch.cat([features[-4],
                                  F.interpolate(features[-3],size=features[-4].size()[-2:]),
                                  F.interpolate(features[-2],size=features[-4].size()[-2:]),
                                  F.interpolate(features[-1],size=features[-4].size()[-2:])],
                                 dim=1
                                 )



        b, c, h, w = gap_features.size()
        seq_deep_features = gap_features.reshape(b, c, -1)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        seq_deep_features = self.MHSA(seq_deep_features, seq_deep_features, seq_deep_features)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        gap_features = seq_deep_features.reshape(b, c, h, w)

        c_last = self.gap(gap_features)

        features[-4:] = self.fpn(features[-4:])

        features[-4:] = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


class Unet_wTri(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.gap = GlobalAvgPool2D()
        # self.feature_latent2048 = nn.Conv2d(self.encoder.out_channels[-1],self.fpn_out_channels,1)
        # self.feature_latent1024 = nn.Conv2d(self.encoder.out_channels[-2],self.fpn_out_channels,1)
        # self.feature_latent512 = nn.Conv2d(self.encoder.out_channels[-3],self.fpn_out_channels,1)

        self.feature_latent2048 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 1)
        self.feature_latent1024 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 1)
        self.feature_latent512 = nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels, 1)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels * 4,1,1),
            nn.Sigmoid()
        )
        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = UnetDecoder(
            encoder_channels=new_encoder_channels,
            decoder_channels= decoder_channels ,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        c_last = self.gap(features[-1])

        features[-4:] = self.fpn(features[-4:])

        features[-4:] = self.sr(c_last, features[-4:])

        # feature2048 = F.interpolate(self.feature_latent2048(features[-1]),features[-4].size()[-2:])
        # feature1024 = F.interpolate(self.feature_latent1024(features[-2]),features[-4].size()[-2:])
        # feature512 = F.interpolate(self.feature_latent512(features[-3]),features[-4].size()[-2:])
        #

        feature2048 = F.interpolate(features[-1],features[-4].size()[-2:])
        feature1024 = F.interpolate(features[-2],features[-4].size()[-2:])
        feature512 = F.interpolate(features[-3],features[-4].size()[-2:])

        spatial_weight = self.spatial_attention(torch.cat([features[-4],feature2048,feature1024,feature512],dim=1))

        #学习空间上的场景关系
        feature2048_weight = F.interpolate(spatial_weight,features[-1].size()[-2:])
        feature1024_weight = F.interpolate(spatial_weight,features[-2].size()[-2:])
        feature512_weight = F.interpolate(spatial_weight,features[-3].size()[-2:])
        features[-1] = features[-1] + features[-1] * feature2048_weight
        features[-2] = features[-2] + features[-2] * feature1024_weight
        features[-3] = features[-3] + features[-3] * feature512_weight
        features[-4] = features[-4] + features[-4] * spatial_weight



        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


class Unet_wTri_wLightDecoder(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.gap = GlobalAvgPool2D()
        self.feature_latent2048 = nn.Conv2d(self.encoder.out_channels[-1],self.fpn_out_channels,1)
        self.feature_latent1024 = nn.Conv2d(self.encoder.out_channels[-2],self.fpn_out_channels,1)
        self.feature_latent512 = nn.Conv2d(self.encoder.out_channels[-3],self.fpn_out_channels,1)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels * 4,1,1),
            nn.Sigmoid()
        )
        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = Light_Decoder(
            encoder_channels=new_encoder_channels,
            decoder_channels=self.fpn_out_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            # in_channels=decoder_channels[-1],
            in_channels=int(self.fpn_out_channels / 2),
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=4
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])

        feature2048 = F.interpolate(self.feature_latent2048(features[-1]),features[-4].size()[-2:])
        feature1024 = F.interpolate(self.feature_latent1024(features[-2]),features[-4].size()[-2:])
        feature512 = F.interpolate(self.feature_latent512(features[-3]),features[-4].size()[-2:])

        spatial_weight = self.spatial_attention(torch.cat([features[-4],feature2048,feature1024,feature512],dim=1))

        features[-4:] = self.fpn(features[-4:])

        #学习空间上的场景关系
        feature2048_weight = F.interpolate(spatial_weight,features[-1].size()[-2:])
        feature1024_weight = F.interpolate(spatial_weight,features[-2].size()[-2:])
        feature512_weight = F.interpolate(spatial_weight,features[-3].size()[-2:])
        features[-1] = features[-1] + features[-1] * feature2048_weight
        features[-2] = features[-2] + features[-2] * feature1024_weight
        features[-3] = features[-3] + features[-3] * feature512_weight
        features[-4] = features[-4] + features[-4] * spatial_weight


        features[-4:] = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


class Unet_wFPN_wSpatial(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained = False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # --------------
        # self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        fpn_in_channels_list = fpn_in_channels_list[-4:]
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        new_encoder_channels = list(self.encoder.out_channels)
        new_encoder_channels[-len(self.encoder_fpn_out_channels):] = self.encoder_fpn_out_channels

        self.gap = GlobalAvgPool2D()
        self.feature_latent2048 = nn.Conv2d(self.encoder.out_channels[-1],self.fpn_out_channels,1)
        self.feature_latent1024 = nn.Conv2d(self.encoder.out_channels[-2],self.fpn_out_channels,1)
        self.feature_latent512 = nn.Conv2d(self.encoder.out_channels[-3],self.fpn_out_channels,1)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels * 4,1,1),
            nn.Sigmoid()
        )
        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels[-4:],
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )

        self.decoder = UnetDecoder(
            encoder_channels=new_encoder_channels,
            decoder_channels= decoder_channels ,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )


        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )


        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)

        # c_last = self.gap(features[-1])

        feature2048 = F.interpolate(self.feature_latent2048(features[-1]),features[-4].size()[-2:])
        feature1024 = F.interpolate(self.feature_latent1024(features[-2]),features[-4].size()[-2:])
        feature512 = F.interpolate(self.feature_latent512(features[-3]),features[-4].size()[-2:])

        spatial_weight = self.spatial_attention(torch.cat([features[-4],feature2048,feature1024,feature512],dim=1))

        features[-4:] = self.fpn(features[-4:])

        #学习空间上的场景关系
        feature2048_weight = F.interpolate(spatial_weight,features[-1].size()[-2:])
        feature1024_weight = F.interpolate(spatial_weight,features[-2].size()[-2:])
        feature512_weight = F.interpolate(spatial_weight,features[-3].size()[-2:])
        features[-1] = features[-1] + features[-1] * feature2048_weight
        features[-2] = features[-2] + features[-2] * feature1024_weight
        features[-3] = features[-3] + features[-3] * feature512_weight
        features[-4] = features[-4] + features[-4] * spatial_weight


        # features[-4:] = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks





if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    data = torch.randn(4,3,256,256)
    # backbone='efficientnet-b0'
    backbone='mobileone_s0'
    # backbone='resnet50'
    # backbone='mit_b0'
    # backbone='mit_b2'
    # backbone='efficientnet-b0'
    # backbone='densenet161'
    in_chns = 3
    class_num1 = 5
    encoder_depth = 5
    decoder_channels = (256, 128, 64, 32, 16)

    # encoder_depth = 4
    # decoder_channels = (256, 128, 64, 32)

    model = Unet_wFPN_wPyramidMHSA_SR_wLightDecoder(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=in_chns,
        fpn_out_channels = 256,
        encoder_depth = encoder_depth,
        decoder_channels = decoder_channels,
        classes=class_num1,
    )
    out = model(data)
    print(out.shape)