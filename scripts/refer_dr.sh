
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/UNet-softmax_focal_blv \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model UNet \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/UNet-softmax_focal_blv/ema-lr6e-4_poly-v2-interval-30/imgz1440_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.8064_iter_3483.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \


Unet_wFPN_wSKA
Unet_wFPN


#
ddr

python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/UNet-softmax_focal_blv \
        --dataset_name DDR \
        --image_size 1024 \
        --model UNet \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/DDR/UNet-softmax_focal_blv/ema-lr6e-4_poly-v2-interval-30/imgz1440_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.6371_iter_3672.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \
