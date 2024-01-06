python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID_smp_UNet_ex/resnet50_shuffle/imgz1440_bs1_Adam_CLAHE2_warmup0_newschduler_max_iterations-4000_polyv2_lr2e-3 \
        --dataset_name crop_IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --optim Adam \
        --batch_size 1 \
        --warmup 0.0 \
        --base_lr 0.002 \
        --CLAHE 2 \
        --ce_weight 1 1 1 1 1 \
        --max_iterations 4000 \
        --backbone resnet50


python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 1 \
        --exp cnn_preprocess_IDRID_smp_UNet_ex/resnet50_shuffle/imgz1440_bs1_Adam_CLAHE2_warmup0_newschduler_max_iterations-7000_polyv2_lr2e-3 \
        --dataset_name cnn_preprocess_IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --optim Adam \
        --batch_size 1 \
        --warmup 0.0 \
        --base_lr 0.002 \
        --CLAHE 0 \
        --ce_weight 1 1 1 1 1 \
        --max_iterations 7000 \
        --backbone resnet50