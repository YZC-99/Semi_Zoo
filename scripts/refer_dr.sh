
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Unet_wFPN \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Unet_wFPN \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/Unet_wFPN/ema-lr6e-4_poly-v2-interval-30/imgz1440_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX0.8106_iter_4698.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \

Unet_wFPN_wSKA
Unet_wFPN


#
ddr

python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Unet_wFPN_wSR-se_resnet50 \
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wFPN_wSR \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/DDR/Unet_wFPN_wSR/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.6623_iter_14796.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \
