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
        --exp crop_IDRID/Unet_wFPN_wASPP_Bottle/ema-lr6e-4_poly-v2-interval-30/imgz1440_se_resnet50/bs2_Adam_CLAHE0-5k\
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Unet_wFPN_wASPP_Bottle \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --decoder_attention_type scse \
        --scheduler poly-v2 \
        --max_iterations 5000 \
        --backbone se_resnet50


        --cutmix_prob 0.5 \

crop_IDRID

        --main_criteria softmax_focal_blv \
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
        --decoder_attention_type scse \


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 1 \
        --exp DDR/Unet_wFPN_wSKA/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wFPN_wSKA \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --backbone se_resnet50





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