


#1、 UNet
python train_supervised_2d_odoc_vessel.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_Unet_only-pseudo-cover-minus-od-rim100/resnet50/imgz256_bs8_Adam_warmup0_iterations-4000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 4000         \
    --backbone resnet50 \
    --vessel_type minus-od-rim100 \
    --autodl

minus-od-rim100
