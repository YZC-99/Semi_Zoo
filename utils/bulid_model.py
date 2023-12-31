import torch
import segmentation_models_pytorch as smp
from model.netwotks.sr_unet import SR_Unet,SR_Unet_woFPN,SR_Unet_SR_FPN,SR_Unet_woSR
from model.netwotks.sr_light_net import LightNet_wFPN,LightNet_wSR,LightNet_wFPN_wSR
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def build_model(args,model,backbone,in_chns,class_num1,class_num2,fuse_type,ckpt_weight=None):
    # scse
    if model == "UNet":
        net =  smp.Unet(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1,
            decoder_attention_type = args.decoder_attention_type
        )
    elif model == 'PAN':
        net =  smp.PAN(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1,
        )
    elif model == 'MAnet':
        net =  smp.MAnet(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1
        )
    elif model == 'DeepLabV3p':
        net =  smp.DeepLabV3Plus(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1
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
    elif model == 'SR_Unet_SR_FPN':
        net =  SR_Unet_SR_FPN(
            encoder_name = backbone,
            encoder_weights = 'imagenet',
            in_channels = in_chns,
            classes= class_num1,
            fpn_out_channels = args.fpn_out_c,
            sr_out_channels=args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type
        )
    elif model == 'SR_Unet_woFPN':
        net = SR_Unet_woFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            sr_out_channels = args.sr_out_c,
            decoder_attention_type =  args.decoder_attention_type
        )

    elif model == 'SR_Unet_woSR':
        net = SR_Unet_woSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels = args.fpn_out_c,
            decoder_attention_type =  args.decoder_attention_type
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




    if ckpt_weight is not None:
        df = torch.load(ckpt_weight,map_location='cpu')
        net.load_state_dict(df)
        print("===================================")
        print("成功加载权重:{}".format(ckpt_weight))
        print("===================================")
    return net
