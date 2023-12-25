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
        self.mean_encoder_channel = int(sum(self.encoder.out_channels) / len(self.encoder.out_channels))
        fpn_in_channels_list = self.encoder.out_channels
        fpn_in_channels_list = [i for i in fpn_in_channels_list if i != 0]
        if len(fpn_in_channels_list) != len(self.encoder.out_channels):
            self.zero_layer = True

        self.gap = GlobalAvgPool2D()
        self.fpn = fpn.FPN(
                            in_channels_list=fpn_in_channels_list,
                            out_channels=self.mean_encoder_channel,
                            conv_block=fpn.default_conv_block,
                            top_blocks=None,)
        self.sr_out_channels = [self.mean_encoder_channel for i in range(len(fpn_in_channels_list))]
        self.sr = SceneRelation(
                            in_channels=self.encoder.out_channels[-1],
                            channel_list=self.sr_out_channels,
                            out_channels=self.mean_encoder_channel,
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

if __name__ == '__main__':
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    data = torch.randn(4,3,256,256)
    # backbone='resnet34'
    # backbone='mit_b0'
    # backbone='mit_b2'
    backbone='efficientnet-b0'
    # backbone='densenet161'
    in_chns = 3
    class_num1 = 5

    model = SR_Unet(
        encoder_name=backbone,
        encoder_weights='imagenet',
        in_channels=in_chns,
        classes=class_num1,
    )
    out = model(data)
    print(out.shape)