#231231
#backbone
nohup python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID_smp_UNet_ex/resnet50/imgz1024_bs2_CLAHE2_lr4e-4 \
        --dataset_name crop_IDRID \
        --val_period 27 \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0004 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50 &

        efficientnet-b0
        mobileone_s3
        resnet50
        se_resnet50
        vgg16
        tu-efficientnet_b0
        timm-efficientnet-b0

# batch size
nohup python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 1 \
        --exp crop_IDRID_smp_UNet_ex/efficientnet-b0_shuffle/imgz1024_bs2_Adam_CLAHE2_warmup0_newschduler_max_iterations-15000 \
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --warmup 0.0 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --max_iterations 15000 \
        --backbone efficientnet-b0 &

# cnnpreprocess
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 1 \
        --exp cnn_preprocess_IDRID_smp_UNet/resnet50_shuffle/imgz800_bs4_Adam \
        --dataset_name cnn_preprocess_IDRID \
        --val_period 54 \
        --image_size 800 \
        --model UNet \
        --optim Adam \
        --batch_size 4 \
        --base_lr 0.0004 \
        --CLAHE 0 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50

#SR
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 1 \
        --exp crop_IDRID_smp_SR_Unet_woFPN_ex/resnet50/imgz1440_CLAHE2_lr2e-4 \
        --dataset_name crop_IDRID \
        --val_period 27 \
        --image_size 1440 \
        --model SR_Unet_woFPN \
        --optim Adam \
        --batch_size 1 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50

# SR_Unet_SR_FPN
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID_smp_SR_Unet_SR_FPN_ex/resnet50/imgz1440_CLAHE2_lr2e-4 \
        --dataset_name crop_IDRID \
        --val_period 27 \
        --image_size 1440 \
        --model SR_Unet_SR_FPN \
        --optim Adam \
        --batch_size 1 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50