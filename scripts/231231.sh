python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID/Unet_wFPN/resnet50/imgz1024_bs2_CLAHE2_lr2e-4_3k \
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model Unet_wFPN \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --autodl \
        --max_iterations 3000 \
        --backbone resnet50

DeepLabv3p
UNetpp
Unet_wRTFM
Unet_wRTFM_wFPN
Unet_wFPN
Unet_wFPN_wDAB
Unet_wRTFM_wFPN_wlightDecoder
Unet_wFPN_wlightDecoder
  mobileone_s3
  efficientnet-b0