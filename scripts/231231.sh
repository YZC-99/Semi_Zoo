python train_idrid_supervised_2d_smp.py \
        --num_works 4 \
        --device 0 \
        --exp crop_IDRID/UNet/resnet50/imgz1024_bs2_CLAHE2_lr2e-4 \
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model UNet \
        --optim Adam \
        --batch_size 2 \
        --base_lr 0.0002 \
        --CLAHE 2 \
        --autodl \
        --backbone resnet50

