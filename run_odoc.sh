python train_odoc_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp RIM-ONE-UNet/resnet50/imgz512_bs8_Adam_warmup0_iterations-4000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 512         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 4000         \
    --autodl         \
    --backbone resnet50

