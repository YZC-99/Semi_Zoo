
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Dual_Decoder_Unet/obj_loss1e-1 \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Dual_Decoder_Unet \
        --CLAHE 0 \
        --autodl \
        --fpn_out_c 48 \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/Dual_Decoder_Unet/obj_loss1e-1_lr6e-4_poly-v2-interval-30/imgz1440_mobileone_s0/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.811_iter_4617.pth \
        --backbone mobileone_s0

Unet_wFPN_wPyramidMHSA_SR
Dual_Decoder_Unet
        --decoder_attention_type scse \
DeepLabV3P
efficientnet-b0
Unet_wFPN_wSR
Unet_wFPN_wSKA
Unet_wFPN
Unet_wFPN_wSpatial
Unet_wTri_wLightDecoder
Unet_wTri
#
ddr

python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Unet_wFPN_wSR \
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wFPN_wSR \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/DDR/Unet_wFPN_wSR/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.6623_iter_14796.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \



python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Unet_wFPN_wSR \
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Unet_wFPN_wSR \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/E-ophtha/Unet_wFPN_wSR/ema-lr6e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k/version0/best_AUC_PR_EX_0.6182_iter_1026.pth \
        --backbone se_resnet50

        --decoder_attention_type scse \
