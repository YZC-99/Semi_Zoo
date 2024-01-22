from typing import Optional, Union, List
import torch
from model.module import fpn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from model.netwotks.unet_decoder import UnetDecoder
from model.compare_modules.rtfm import RTFM

class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
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

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
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

    def forward_logits(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return decoder_output

    def forward_seg(self, logits):

        masks = self.segmentation_head(logits)

        return masks

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class Unet_wRTFM(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
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

        self.RTFM = RTFM(self.encoder.out_channels[-1],192,192)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
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

    def forward_logits(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return decoder_output

    def forward_seg(self, logits):

        masks = self.segmentation_head(logits)

        return masks

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)


        deep_features = features[-1]
        b,c,h,w = deep_features.size()
        seq_deep_features = deep_features.reshape(b,c,-1)
        seq_deep_features = seq_deep_features.permute(0,2,1)
        seq_deep_features = self.RTFM(seq_deep_features,seq_deep_features,seq_deep_features)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        features[-1] = seq_deep_features.reshape(b,c,h,w)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks



class Two_Encoder_Unet_wRTFM(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder2_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
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
        self.encoder2 = get_encoder(
            encoder2_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=None,
        )

        self.RTFM = RTFM(self.encoder.out_channels[-1],192,192)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
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

    def forward_logits(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        return decoder_output

    def forward_seg(self, logits):

        masks = self.segmentation_head(logits)

        return masks

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        features2 = self.encoder2(x)


        deep_features = features[-1]
        b,c,h,w = deep_features.size()
        seq_deep_features = deep_features.reshape(b,c,-1)
        seq_deep_features = seq_deep_features.permute(0,2,1)

        deep_features2 = features2[-1]
        b,c,h,w = deep_features2.size()
        seq_deep_features2 = deep_features2.reshape(b,c,-1)
        seq_deep_features2 = seq_deep_features2.permute(0,2,1)

        seq_deep_features = self.RTFM(seq_deep_features,seq_deep_features,seq_deep_features2)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        features[-1] = seq_deep_features.reshape(b,c,h,w)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


class Unet_wFPN_wRTFM(SegmentationModel):

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

        # --------------
        # if self.zero_layer:
        #     self.sr_out_channels.insert(1, 0)


        self.RTFM = RTFM(self.encoder.out_channels[-1],64,64)


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

    def forward_logits(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)

        fpn_features = self.fpn(features)

        deep_features = fpn_features[-1]
        b,c,h,w = deep_features.size()
        seq_deep_features = deep_features.reshape(b,c,-1)
        seq_deep_features = seq_deep_features.permute(0,2,1)
        seq_deep_features = self.RTFM(seq_deep_features,seq_deep_features,seq_deep_features)
        seq_deep_features = seq_deep_features.permute(0, 2, 1)
        fpn_features[-1] = seq_deep_features.reshape(b,c,h,w)


        decoder_output = self.decoder(*fpn_features)

        return decoder_output

    def forward_seg(self, logits):

        masks = self.segmentation_head(logits)

        return masks


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

    data = torch.randn(4, 3, 256, 256)
    backbone = 'resnet50'
    # backbone='mit_b0'
    # backbone='mit_b2'
    # backbone='efficientnet-b0'
    # backbone='densenet161'
    in_chns = 3
    class_num1 = 5
    #
    model = Unet_wFPN_wRTFM(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=in_chns,
        classes=class_num1,
    )
    out = model(data)
    print(out.shape)