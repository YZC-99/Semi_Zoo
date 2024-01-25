python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID/Dual_Decoder_Unet_wFPN_wDAB/resnet50/nolrdecay/imgz1024_bs2_Adam_CLAHE0_lr3e-4_5k\
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wFPN_wDAB \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0003 \
        --CLAHE 0 \
        --autodl \
        --scheduler no\
        --max_iterations 5000 \
        --obj_loss 1.0 \
        --backbone resnet50

        --main_criteria softmax_focal \

        --obj_loss 1.0 \

DeepLabv3p
UNetpp
Unet_wRTFM
Unet_wRTFM_wFPN
Unet_wFPN
Unet_wDAB

Unet_wFPN_wDAB
Unet_wFPN_wSR
Unet_wRTFM_wFPN_wlightDecoder
Unet_wFPN_wlightDecoder
Dual_Decoder_Unet
Dual_Seg_Head_Unet
  mobileone_s3
  efficientnet-b0