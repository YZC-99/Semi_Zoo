python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE-UNet/mobileone_s3/imgz256_bs8_Adam_warmup0_iterations-2000_polyv2_lr1e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0001         \
    --max_iterations 2000         \
    --backbone mobileone_s3 \
    --autodl






python train_odoc_supervised_2d_smp_aux.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE-Dual_Decoder_Unet-aux/resnet50/vessel_loss_weight1e_imgz256_bs16_Adam_warmup0_iterations-2000_polyv2_lr1e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 16         \
    --labeled_bs 8         \
    --warmup 0.0         \
    --base_lr 0.0001         \
    --max_iterations 2000         \
    --backbone resnet50 \
    --vessel_loss_weight 1.0 \
    --autodl


