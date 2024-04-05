python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/UNet/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 1200 \
        --backbone mobileone_s0


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/UNet-softmax_focal/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --main_criteria softmax_focal \
        --max_iterations 1200 \
        --backbone mobileone_s0


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Unet_wPyramidAttentionASPP_wMain/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Unet_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 1200 \
        --backbone mobileone_s0

python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Unet_wMSFE_wPyramidAttentionASPP_wMain/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --max_iterations 1200 \
        --backbone mobileone_s0


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --obj_loss 1.0 \
        --max_iterations 1200 \
        --backbone mobileone_s0


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp E-ophtha/Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain-softmax_focal/lr6e-4_poly-v2-interval-30/imgz1024_mobileone_s0/bs2_Adam_CLAHE0-5k\
        --dataset_name E-ophtha \
        --image_size 1024 \
        --model Dual_Decoder_Unet_wMSFE_wPyramidAttentionASPP_wMain \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0006 \
        --CLAHE 0 \
        --save_period 1200 \
        --autodl \
        --ema 0.75 \
        --scheduler poly-v2 \
        --obj_loss 1.0 \
        --main_criteria softmax_focal \
        --max_iterations 1200 \
        --seed 3407 \
        --backbone mobileone_s0