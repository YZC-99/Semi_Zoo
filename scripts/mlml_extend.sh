python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Unet_wMSFE_wFPN_wDeocderAttention-softmax_focal_blv/lr6e-4_imgz1440_se_resnet50 \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Unet_wMSFE_wFPN_wDeocderAttention \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --fpn_out_c 48 \
        --scheduler poly-v2 \
        --max_iterations 5000 \
        --main_criteria softmax_focal_blv \
        --encoder_deepth 5 \
        --backbone se_resnet50



Unet_wMSFE_wFPN_wDeocderAttention
Unet_wASPP_wFPN_wDeocderAttention
Unet_wASPPv2_wFPN_wDeocderAttention