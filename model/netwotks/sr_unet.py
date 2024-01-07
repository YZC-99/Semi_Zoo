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
from model.module.farseg import SceneRelation

class SR_Unet(SegmentationModel):


    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        fpn_out_channels = -1,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        #--------------
        self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = fpn_out_channels
        

        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        if len(fpn_in_channels_list) != len(self.encoder.out_channels):
            self.zero_layer = True

        self.gap = GlobalAvgPool2D()
        self.fpn = fpn.FPN(
                            in_channels_list=fpn_in_channels_list,
                            out_channels=self.fpn_out_channels,
                            conv_block=fpn.default_conv_block,
                            top_blocks=None,)
        self.sr_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels,
                            out_channels=self.fpn_out_channels,
                            scale_aware_proj=True,
                        )
        #--------------
        if self.zero_layer:
            self.sr_out_channels.insert(1,0)

        self.decoder = UnetDecoder(
            encoder_channels=self.sr_out_channels,
            decoder_channels=decoder_channels,
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
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])
        if self.zero_layer:
            org_feature = features.pop(1)

        fpn_features = self.fpn(features)

        refined_fpn_feature = self.sr(c_last, fpn_features)

        if self.zero_layer:
            refined_fpn_feature_list = list(refined_fpn_feature)
            refined_fpn_feature_list.insert(1,org_feature)
            refined_fpn_feature = tuple(refined_fpn_feature_list)


        decoder_output = self.decoder(*refined_fpn_feature)



        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks




class SR_Unet_SR_FPN(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            sr_out_channels = 128,
            fpn_out_channels = -1,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )


        self.sr_out_channels = sr_out_channels
        self.sr_out_channels_list = [self.sr_out_channels for i in range(len(self.encoder.out_channels))]

        self.gap = GlobalAvgPool2D()

        self.sr = SceneRelation(
            in_channels=self.encoder.out_channels[-1],
            channel_list=self.encoder.out_channels,
            out_channels=self.sr_out_channels,
            scale_aware_proj=True,
        )
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
        self.zero_layer = False
        if fpn_out_channels < 0:
            self.fpn_out_channels = int(sum(self.sr_out_channels_list) / len(self.encoder.out_channels))
        else:
            self.fpn_out_channels = self.sr_out_channels_list
        fpn_in_channels_list = [self.sr_out_channels for i in range(len(self.encoder.out_channels))]
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        if len(fpn_in_channels_list) != len(self.encoder.out_channels):
            self.zero_layer = True

        self.fpn = fpn.FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )
        # Update SceneRelation weights
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

        self.decoder = UnetDecoder(
            encoder_channels=fpn_in_channels_list,
            decoder_channels=decoder_channels,
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
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])


        refined_feature = self.sr(c_last, features)



        if self.zero_layer:
            org_feature = refined_feature.pop(1)

        fpn_refined_feature = self.fpn(refined_feature)


        if self.zero_layer:
            fpn_refined_feature_list = list(fpn_refined_feature)
            fpn_refined_feature_list.insert(1, org_feature)
            fpn_refined_feature = tuple(fpn_refined_feature_list)

        decoder_output = self.decoder(*fpn_refined_feature)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks




class SR_Unet_woFPN(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            sr_out_channels=128,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
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
        self.sr_out_channels_list = [self.sr_out_channels for i in range(len(self.encoder.out_channels))]

        self.gap = GlobalAvgPool2D()

        self.sr = SceneRelation(
            in_channels=self.encoder.out_channels[-1],
            channel_list=self.encoder.out_channels,
            out_channels=self.sr_out_channels,
            scale_aware_proj=True,
        )
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

        self.decoder = UnetDecoder(
            encoder_channels=self.sr_out_channels_list,
            decoder_channels=decoder_channels,
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
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        c_last = self.gap(features[-1])


        refined_feature = self.sr(c_last, features)

        decoder_output = self.decoder(*refined_feature)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class SR_Unet_woSR(SegmentationModel):

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
            in_channels_list=fpn_in_channels_list,
            out_channels=self.fpn_out_channels,
            conv_block=fpn.default_conv_block,
            top_blocks=None, )

        self.encoder_fpn_out_channels = [self.fpn_out_channels for i in range(len(fpn_in_channels_list))]

        # --------------
        # if self.zero_layer:
        #     self.sr_out_channels.insert(1, 0)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder_fpn_out_channels,
            decoder_channels=decoder_channels,
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
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        fpn_features = self.fpn(features)

        decoder_output = self.decoder(*fpn_features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    data = torch.randn(4,3,256,256)
    backbone='resnet34'
    # backbone='mit_b0'
    # backbone='mit_b2'
    # backbone='efficientnet-b0'
    # backbone='densenet161'
    in_chns = 3
    class_num1 = 5

    model = SR_Unet_woSR(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=in_chns,
        classes=class_num1,
    )
    out = model(data)
    print(out.shape)