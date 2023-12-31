#231231
python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 1 \
        --exp crop_IDRID_smp_UNet_ex/imgz1024_bs2_CLAHE2_lr4e-4 \
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0004 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50


python train_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 0 \
        --exp crop_IDRID_smp_SR_UNet_ex/imgz1024_bs2_CLAHE2_lr4e-4_c32 \
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1024 \
        --model SR_Unet \
        --fpn_out_c 32 \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0004 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50