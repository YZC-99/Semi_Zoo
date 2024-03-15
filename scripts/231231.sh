#python train_idrid_supervised_2d_smp.py \
#        --num_works 4 \
#        --device 0 \
#        --exp crop_IDRID/Unet_wFPN-wDecoder-scpsa/resnet50/imgz800_bs4_Adam_CLAHE0_lr6e-4-5k\
#        --dataset_name crop_IDRID \
#        --image_size 800 \
#        --model Unet_wFPN \
#        --optim Adam \
#        --batch_size 4 \
#        --base_lr 0.0006 \
#        --CLAHE 0 \
#        --autodl \
#        --scheduler no \
#        --max_iterations 5000 \
#        --backbone resnet50
Unet_wFPN_wSKA_add_Spatial

python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Unet_wFPN/fpn128-lr6e-4_poly-v2-interval-30/imgz1440_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Unet_wFPN \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --fpn_out_c 128 \
        --scheduler poly-v2 \
        --max_iterations 5000 \
        --encoder_deepth 5 \
        --backbone mobileone_s0

Unet_wFPN_wPyramidASPP
Unet_wFPN_wDouble_ASPP
Unet_wFPN_wASPP
Unet_wFPN_wDouble_ASPP
12head-128dim-
Unet_wFPN_wPyramidMHSA_SR_wLightDecoder
Unet_wFPN_wPyramidMHSA_SR
Unet_wTri_wLightDecoder
fpn_out_c
mobileone_s0
mobileone_s0
efficientnet-b0 24


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP_wMain/fpn48-obj_loss1e-1_lr6e-4_poly-v2-interval-30/imgz1440_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 5000 \
        --fpn_out_c 48 \
        --obj_loss 0.1 \
        --backbone mobileone_s0

        --main_criteria softmax_focal_blv \

Dual_Decoder_Unet_wAuxInPyramidASPP_wMain
Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP_wMain
Unet_wFPN_wSR
Unet_wTri
        --cutmix_prob 0.5 \
Dual_Decoder_Unet_wFPN_wAuxInPyramidASPP
crop_IDRID
Unet_wFPN_wSpatial
        DeepLabV3P
Unet_wFPN_wASPP_Bottle
annealing_softmax_focal_blv
Unet_wRTFM_wFPN
softmax_focal_blv
Unet_wMHSA_wFPN
Unet_wFPN
Unet_wFPN_wSKA_Dali
Unet_wFPN_wSKA
Unet_wFPN_wSCCBAM
Unet_wMamba_Bot
Unet_wDeocderAttention
        --decoder_attention_type scse \


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Unet_wFPN_wDeocderAttention/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wFPN_wDeocderAttention \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --decoder_attention_type scse \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --backbone se_resnet50


        --main_criteria softmax_focal_blv \




DeepLabv3p
UNetpp
Unet_wRTFM
Unet_wRTFM_wFPN
Unet_wFPN
Unet_wDAB
Unet_wFPN_wDAB_wSR
Unet_wFPN_wDAB
Unet_wFPN_wSR
Unet_wRTFM_wFPN_wlightDecoder
Unet_wFPN_wlightDecoder
Dual_Decoder_Unet
Dual_Decoder_Unet_wFPN_wDAB
Dual_Seg_Head_Unet
  mobileone_s3
  efficientnet-b0



python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Unet_wFPN_wSR/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Unet_wFPN_wSR \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 1200 \
        --backbone se_resnet50

        --decoder_attention_type scse \

        --main_criteria softmax_focal_blv \

Unet_wDeocderAttention
Unet_wFPN_wDeocderAttention