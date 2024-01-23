python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID/UNet/resnet50-deepth3/imgz1024_bs2_CLAHE2_lr2e-4_5k \
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model UNet \
        --encoder_deepth 3 \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --autodl \
        --max_iterations 5000 \
        --backbone resnet50

Unet_wRTFM
Unet_wRTFM_wFPN
Unet_wFPN
Unet_wFPN_wDAB
Unet_wRTFM_wFPN_wlightDecoder
Unet_wFPN_wlightDecoder
  mobileone_s3
  efficientnet-b0