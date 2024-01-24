python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID/Unet_wRTFM/resnet50_qkv/imgz1024_bs2_CLAHE2_lr2e-4_3k \
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model Unet_wRTFM \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --autodl \
        --max_iterations 3000 \
        --backbone resnet50


        --obj_loss 1.0 \

DeepLabv3p
UNetpp
Unet_wRTFM
Unet_wRTFM_wFPN
Unet_wFPN
Unet_wFPN_wDAB
Unet_wFPN_wSR
Unet_wRTFM_wFPN_wlightDecoder
Unet_wFPN_wlightDecoder
Dual_Decoder_Unet
Dual_Seg_Head_Unet
  mobileone_s3
  efficientnet-b0