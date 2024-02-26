
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/UNetpp \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model UNetpp \
        --CLAHE 0 \
        --autodl \
        --decoder_attention_type scse \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/UNetpp/ema-lr6e-4_poly-v2-interval-30/imgz1440_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.8148_iter_4995.pth \
        --backbone se_resnet50


Unet_wFPN_wSKA
Unet_wFPN


#
ddr

python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 1 \
        --exp DDR/Unet_wFPN_wSKA-se_resnet50 \
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wFPN_wSKA \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/DDR/Unet_wFPN_wSKA/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX0.5693_iter_4482.pth \
        --backbone se_resnet50

