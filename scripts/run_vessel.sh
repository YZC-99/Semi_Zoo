


#1、 UNet
python train_supervised_2d_odoc_vessel.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE_only-pseudo-cover-oc-rim-5p/UNet/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --save_period 500 \
    --autodl


