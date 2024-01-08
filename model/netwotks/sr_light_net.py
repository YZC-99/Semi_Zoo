from typing import Optional, Union, List
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from model.module.gap import GlobalAvgPool2D
from model.module import fpn
from model.module.light_decoder import Light_Decoder
from model.module.farseg import SceneRelation
import torch.nn as nn



class LightNet_wFPN(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            fpn_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            fpn_pretrained=False,
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
        # if len(fpn_in_channels_list) != len(self.encoder.out_channels):
        #     self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list[-4:],
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        if fpn_pretrained:
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

            self.fpn.load_state_dict(fpn_state_dict, strict=False)
            print("================加载FPN权重成功！===============")
            print(fpn_state_dict.keys())
            # --------------

        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        # --------------
        # if self.zero_layer:
        #     self.sr_out_channels.insert(1, 0)

        self.decoder = Light_Decoder(
            encoder_channels=self.encoder_fpn_out_channels,
            decoder_channels=self.encoder_fpn_out_channels[1:],
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=128,
            out_channels=classes,
            activation=activation,
            upsampling = 0,
            kernel_size=3,
        )

        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        fpn_features = self.fpn(features[-4:])

        decoder_output = self.decoder(*fpn_features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        masks = self.upsample4x_op(masks)

        return masks

class LightNet_wSR(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            sr_out_channels=256,
            decoder_use_batchnorm: bool = True,
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
            sr_pretrained=False,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )


        # --------------

        self.sr_out_channels = sr_out_channels
        self.sr_out_channels_list = [self.sr_out_channels for i in range(len(self.encoder.out_channels[-4:]))]

        self.gap = GlobalAvgPool2D()

        self.sr = SceneRelation(
            in_channels=self.encoder.out_channels[-1],
            channel_list=self.encoder.out_channels[-4:],
            out_channels=self.sr_out_channels,
            scale_aware_proj=True,
        )
        if sr_pretrained:
            # --------------
            ## 加载sr的预训练权重
            ckpt_apth = 'pretrained/farseg50.pth'
            sd = torch.load(ckpt_apth)
            # Update SceneRelation weights
            sr_state_dict = self.sr.state_dict()
            for name, param in sd['model'].items():

                if 'module.sr' in name:
                    # 移除 'module.' 前缀
                    name = name.replace('module.', '')
                    # Update SceneRelation state_dict
                    sr_state_dict[name] = param
            # Load the modified SceneRelation state_dict
            self.sr.load_state_dict(sr_state_dict,strict=False)
            print("================加载SR权重成功！===============")
            print(sr_state_dict.keys())
                # --------------

        # --------------
        # if self.zero_layer:
        #     self.sr_out_channels.insert(1, 0)

        self.decoder = Light_Decoder(
            encoder_channels=self.sr_out_channels_list,
            decoder_channels=1,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=128,
            out_channels=classes,
            activation=activation,
            upsampling = 0,
            kernel_size=3,
        )

        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])

        refined_feature = self.sr(c_last, features[-4:])

        decoder_output = self.decoder(*refined_feature)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        masks = self.upsample4x_op(masks)

        return masks


if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    data = torch.randn(4, 3, 256, 256)
    backbone = 'resnet50'
    # backbone='mit_b0'
    # backbone='mit_b2'
    # backbone='efficientnet-b0'
    # backbone='densenet161'
    in_chns = 3
    class_num1 = 5

    model = LightNet_wSR(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=in_chns,
        classes=class_num1,
    )
    out = model.forward(data)
    print(out.shape)