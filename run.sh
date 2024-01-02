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
#CHALHE
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID_smp_UNet_ex/imgz1024_bs2_CLAHE4032_lr4e-4_resnet50/CHALHE4\
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0004 \
        --CLAHE 4032 \
        --ce_weight 1 1 1 1 1 \
        --backbone resnet50
        
         0,2,2032,4,4032

# batch size
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID_smp_UNet_ex/bs/imgz1440_CLAHE2_lr1e-1_resnet50_SGD_max_iterations2w \
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --optim SGD \
        --batch_size 1 \
        --base_lr 0.1 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --max_iterations 20000 \
        --backbone resnet50

# cnnpreprocess
python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 1 \
        --exp cnn_preprocess_IDRID_smp_UNet/resnet50/imgz1440_bs1_SGD_max_iterations2w \
        --dataset_name cnn_preprocess_IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --optim SGD \
        --batch_size 1 \
        --base_lr 0.1 \
        --CLAHE 0 \
        --ce_weight 1 1 1 1 1 \
        --max_iterations 20000 \
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