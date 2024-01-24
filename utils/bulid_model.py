import torch
import segmentation_models_pytorch as smp
from model.netwotks.unet import Unet,Unet_wRTFM,Unet_wFPN_wlightDecoder,Unet_wRTFM_wFPN_wlightDecoder,Two_Encoder_Unet_wRTFM
from model.netwotks.unet import Unet_wFPN,Unet_wRTFM_wFPN,Unet_wFPN_wDAB
from model.netwotks.sr_unet import SR_Unet,SR_Unet_woFPN,SR_Unet_SR_FPN,SR_Unet_woSR
from model.netwotks.sr_light_net import LightNet_wFPN,LightNet_wSR,LightNet_wFPN_wSR
from model.netwotks.dual_decoer_unet import Dual_Decoder_Unet,Dual_Seg_Head_Unet,Dual_Decoder_SR_Unet,Dual_Decoder_SR_Unet_woSR,Dual_Decoder_SR_Unet_woFPN
from model.netwotks.dual_decoer_unet import Dual_Decoder_Unet_DualFPN_CrossAttention,Dual_Decoder_SR_Unet_woSR_wRTFM
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def build_model(args,model,backbone,in_chns,class_num1,class_num2,fuse_type,ckpt_weight=None):
    decoder_channels = [256, 128, 64, 32, 16]
    decoder_channels = decoder_channels[:args.encoder_deepth]
    # scse
    if model == "UNet":
        net =  Unet(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            encoder_depth=args.encoder_deepth,
            in_channels = in_chns,
            classes= class_num1,
            decoder_attention_type = args.decoder_attention_type,
            decoder_channels = decoder_channels
        )
    elif model == "Unet_wFPN":
        net = Unet_wFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wDAB":
        net = Unet_wFPN_wDAB(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wRTFM":
        net = Unet_wRTFM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wRTFM_wFPN":
        net = Unet_wRTFM_wFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wlightDecoder":
        net = Unet_wFPN_wlightDecoder(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )
    elif model == "Unet_wRTFM_wFPN_wlightDecoder":
        net = Unet_wRTFM_wFPN_wlightDecoder(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )




    elif model == "Unet_wFPN_wRTFM":
        net = Unet_wFPN_wRTFM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )

    elif model == "Two_Encoder_Unet_wRTFM":
        net = Two_Encoder_Unet_wRTFM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )

    elif model == 'SR_Unet':
        net =  SR_Unet(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1,
            fpn_out_channels = args.fpn_out_c,
            decoder_attention_type = args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
            sr_pretrained=args.sr_pretrained
        )
    elif model == 'Dual_Decoder_Unet_DualFPN_CrossAttention':
        net = Dual_Decoder_Unet_DualFPN_CrossAttention(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
        )

    elif model == 'Dual_Decoder_SR_Unet_woSR':
        net = Dual_Decoder_SR_Unet_woSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
        )
    elif model == 'Dual_Decoder_SR_Unet_woSR_wRTFM':
        net = Dual_Decoder_SR_Unet_woSR_wRTFM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
        )


    elif model == 'Dual_Decoder_SR_Unet_woFPN':
        net = Dual_Decoder_SR_Unet_woFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            sr_out_channels=args.sr_out_c,
            decoder_attention_type=args.decoder_attention_type,
            sr_pretrained=args.sr_pretrained,
        )
    elif model == 'SR_Unet_SR_FPN':
        net =  SR_Unet_SR_FPN(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1,
            fpn_out_channels = args.fpn_out_c,
            sr_out_channels=args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            sr_pretrained=args.sr_pretrained,
            fpn_pretrained=args.fpn_pretrained,
        )
    elif model == 'SR_Unet_woFPN':
        net = SR_Unet_woFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            sr_out_channels = args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            sr_pretrained=args.sr_pretrained,
        )
    elif model == 'SR_Unet_woSR':
        net = SR_Unet_woSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels = args.fpn_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
        )
    elif model == 'LightNet_wFPN':
        net = LightNet_wFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels = args.fpn_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained
        )
    elif model == 'LightNet_wSR':
        net = LightNet_wSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            sr_out_channels = args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            sr_pretrained=args.sr_pretrained
        )
    elif model == 'LightNet_wFPN_wSR':
        net = LightNet_wFPN_wSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels = args.fpn_out_c,
            sr_out_channels=args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type,
            fpn_pretrained=args.fpn_pretrained,
            sr_pretrained=args.sr_pretrained
        )

    elif model == 'Dual_Decoder_Unet':
        net = Dual_Decoder_Unet(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
        )
    elif model == 'Dual_Seg_Head_Unet':
        net = Dual_Seg_Head_Unet(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
        )
    elif model == 'Dual_Decoder_SR_Unet':
        net = Dual_Decoder_SR_Unet(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_pretrained=args.fpn_pretrained,
            sr_pretrained=args.sr_pretrained
        )

    elif model == 'DeepLabv3p':
        net = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
        )



    if ckpt_weight is not None:
        if args.exclude_keys is not None:
            checkpoint = torch.load(ckpt_weight, map_location=lambda storage, loc: storage)
            filtered_checkpoint = {key: value for key, value in checkpoint.items() if
                                   all(exclude_key not in key for exclude_key in exclude_keys)}
            net.load_state_dict(filtered_checkpoint, strict = False)
        else:
            checkpoint = torch.load(ckpt_weight, map_location='cpu')
            net.load_state_dict(checkpoint)
        print("===================================")
        print("成功加载权重:{}".format(ckpt_weight))
        print("===================================")
    return net
