# UNet
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/UNet/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --encoder_deepth 5 \
        --backbone mobileone_s0

#UNet-softmaxloss
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/UNet-softmax_focal/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --encoder_deepth 5 \
        --main_criteria softmax_focal \
        --backbone mobileone_s0

#Unet_wPyramidAttentionASPP_wMain
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Unet_wPyramidAttentionASPP_wMain/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --encoder_deepth 5 \
        --backbone mobileone_s0

#Unet_wMSFE_wPyramidAttentionASPP_wMain
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Unet_wMSFE_wPyramidAttentionASPP_wMain/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --encoder_deepth 5 \
        --backbone mobileone_s0

# Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain/obj_loss1e_lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --obj_loss 1.0 \
        --backbone mobileone_s0


# Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain-softmax_focal
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp DDR/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain-softmax_focal/obj_loss1e_lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name DDR \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 15000 \
        --obj_loss 1.0 \
        --main_criteria softmax_focal \
        --backbone mobileone_s0



