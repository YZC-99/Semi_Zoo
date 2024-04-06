
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID/Dual_Unet_wASPPv2_wFPN_wDeocderAttention-softmax_focal \
        --dataset_name crop_IDRID \
        --image_size 1440 \
        --model Dual_Unet_wASPPv2_wFPN_wDeocderAttention \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/Dual_Unet_wASPPv2_wFPN_wDeocderAttention-softmax_focal_blv/lr6e-4_imgz1440_se_resnet50/version01/best_AUC_PR_EX_0.8237_iter_4779.pth \
        --backbone se_resnet50

        --fpn_out_c 48 \

        --decoder_attention_type scse \

Unet_wFPN_wPyramidMHSA_SR
Dual_Decoder_Unet
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
        --exp DDR/UNet \
        --dataset_name DDR \
        --image_size 1024 \
        --model UNet \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/E-ophtha/UNet/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k/version06/best_AUC_PR_EX_0.5196_iter_918.pth \
        --backbone mobileone_s0



        --decoder_attention_type scse \



python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain-softmax_focal \
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/E-ophtha/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain-softmax_focal/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k/version06/best_AUC_PR_EX_0.6206_iter_783.pth \
        --backbone mobileone_s0

