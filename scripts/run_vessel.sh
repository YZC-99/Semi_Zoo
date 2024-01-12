#1、 UNet
python train_vessel_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp DRIVE/UNet/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name DRIVE        \
    --image_size 256         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

