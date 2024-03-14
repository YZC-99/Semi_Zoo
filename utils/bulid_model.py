import torch
import segmentation_models_pytorch as smp
from model.netwotks.unet import Unet_wMamba_Bot,Unet_wFPN_wMamba
from model.netwotks.unet import Unet,Unet_wRTFM,Unet_wFPN_wlightDecoder,Unet_wRTFM_wFPN_wlightDecoder,Two_Encoder_Unet_wRTFM
from model.netwotks.unet import Unet_wFPN,Unet_wRTFM_wFPN,Unet_wFPN_wDAB,Unet_wFPN_wSR,Unet_wDAB,Unet_wFPN_wDAB_wSR
from model.netwotks.unet import Unet_wFPN_wSKA_add_CBAM,Unet_wFPN_wSKA_Dali,Unet_wMHSA_wFPN
from model.netwotks.unet import Unet_wFPN_wASPP_Bottle,Unet_wFPN_wDeocderAttention,Unet_wDeocderAttention
from model.netwotks.unet import Unet_wSR,Unet_wFPN_wDAB_wSR_wRTFM,Unet_wFPN_wSKA,Unet_wFPN_wSCBAM,Unet_wFPN_wSCCBAM,Unet_wFPN_wSKA_add_Spatial
from model.netwotks.att_unet import AttU_Net
from model.netwotks.sr_unet import Unet_wTri,Dual_Decoder_Unet_wTri,Unet_wFPN_wSpatial,Unet_wTri_wLightDecoder
from model.netwotks.sr_unet import Unet_wFPN_wPyramidMHSA_SR_wLightDecoder,Unet_wFPN_wPyramidMHSA_SR,Unet_wFPN_wASPP,Unet_wFPN_wDouble_ASPP
from model.netwotks.sr_unet import Unet_wFPN_wPyramidASPP,Unet_wFPN_wPyramidASPP_inFPN,Dual_Decoder_Unet_wFPN_wPyramidASPP
from model.netwotks.sr_unet import Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP
from model.netwotks.sr_light_net import LightNet_wFPN,LightNet_wSR,LightNet_wFPN_wSR
from model.netwotks.dual_decoer_unet import Dual_Decoder_Unet,Dual_Seg_Head_Unet,Dual_Decoder_SR_Unet,Dual_Decoder_SR_Unet_woSR,Dual_Decoder_SR_Unet_woFPN
from model.netwotks.dual_decoer_unet import Dual_Decoder_Unet_DualFPN_CrossAttention,Dual_Decoder_SR_Unet_woSR_wRTFM
from model.netwotks.dual_decoer_unet import Dual_Decoder_Unet_wFPN_wDAB,Dual_Decoder_Unet_wFPN,Dual_Decoder_Unet_wFPN_wSKA
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def build_model(args,model,backbone,in_chns,class_num1,class_num2,fuse_type,ckpt_weight=None):
    decoder_channels = [256, 128, 64, 32, 16]
    # decoder_channels = decoder_channels[:args.encoder_deepth]
    decoder_channels = decoder_channels[-args.encoder_deepth:]
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


    elif model == "Unet_wDeocderAttention":
        net = Unet_wDeocderAttention(
            encoder_name=backbone,
            encoder_weights='imagenet',
            encoder_depth=args.encoder_deepth,
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wMamba_Bot":
        net = Unet_wMamba_Bot(
            encoder_name=backbone,
            encoder_weights='imagenet',
            encoder_depth=args.encoder_deepth,
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            decoder_channels=decoder_channels
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
    elif model == "Unet_wFPN_wPyramidASPP":
        net = Unet_wFPN_wPyramidASPP(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wASPP":
        net = Unet_wFPN_wASPP(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wPyramidASPP_inFPN":
        net = Unet_wFPN_wPyramidASPP_inFPN(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Dual_Decoder_Unet_wFPN_wPyramidASPP":
        net = Dual_Decoder_Unet_wFPN_wPyramidASPP(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP":
        net = Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wDouble_ASPP":
        net = Unet_wFPN_wDouble_ASPP(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )

    elif model == "Unet_wFPN_wDeocderAttention":
        net = Unet_wFPN_wDeocderAttention(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wASPP_Bottle":
        net = Unet_wFPN_wASPP_Bottle(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wSR":
        net = Unet_wSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )

    elif model == "Unet_wDAB":
        net = Unet_wDAB(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wMamba":
        net = Unet_wFPN_wMamba(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wSKA_Dali":
        net = Unet_wFPN_wSKA_Dali(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )

    elif model == "Unet_wFPN_wSKA":
        net = Unet_wFPN_wSKA(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wSKA_add_Spatial":
        net = Unet_wFPN_wSKA_add_Spatial(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wSKA_add_CBAM":
        net = Unet_wFPN_wSKA_add_CBAM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )



    elif model == "Unet_wFPN_wSCBAM":
        net = Unet_wFPN_wSCBAM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wSCCBAM":
        net = Unet_wFPN_wSCCBAM(
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
    elif model == "Unet_wFPN_wDAB_wSR":
        net = Unet_wFPN_wDAB_wSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wDAB_wSR_wRTFM":
        net = Unet_wFPN_wDAB_wSR_wRTFM(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wSR":
        net = Unet_wFPN_wSR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wTri":
        net = Unet_wTri(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Dual_Decoder_Unet_wTri":
        net = Dual_Decoder_Unet_wTri(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wPyramidMHSA_SR":
        net = Unet_wFPN_wPyramidMHSA_SR(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )
    elif model == "Unet_wFPN_wPyramidMHSA_SR_wLightDecoder":
        net = Unet_wFPN_wPyramidMHSA_SR_wLightDecoder(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )

    elif model == "Unet_wTri_wLightDecoder":
        net = Unet_wTri_wLightDecoder(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
            decoder_attention_type=args.decoder_attention_type,
            encoder_depth=args.encoder_deepth,
            decoder_channels=decoder_channels
        )


    elif model == "Unet_wFPN_wSpatial":
        net = Unet_wFPN_wSpatial(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            fpn_out_channels=args.fpn_out_c,
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
    elif model == "Unet_wMHSA_wFPN":
        net = Unet_wMHSA_wFPN(
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

    elif model == "Dual_Decoder_Unet_wFPN_wSKA":
        net = Dual_Decoder_Unet_wFPN_wSKA(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )


    elif model == "Dual_Decoder_Unet_wFPN_wDAB":
        net = Dual_Decoder_Unet_wFPN_wDAB(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            decoder_attention_type=args.decoder_attention_type
        )

    elif model == "Dual_Decoder_Unet_wFPN":
        net = Dual_Decoder_Unet_wFPN(
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
            encoder_depth= args.encoder_deepth
        )
    elif model == 'UNetpp':
        net = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            encoder_depth= args.encoder_deepth
        )
    elif model == 'DeepLabV3P':
        net = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights='imagenet',
            in_channels=in_chns,
            classes=class_num1,
            encoder_depth= args.encoder_deepth
        )
    elif model == 'AttU_Net':
        net = AttU_Net(
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
