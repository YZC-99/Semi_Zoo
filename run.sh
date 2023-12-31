python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp patch_IDRID_smp_UNet_ex/resnet50_shuffle/imgz1024_bs4_SGD_CLAHE0_warmup0_newschduler_max_iterations-10000_polyv2_lr7e-3        \
    --dataset_name patch_IDRID         \
    --val_period 54         \
    --image_size 1024         \
    --model UNet         \
    --optim SGD         \
    --batch_size 4         \
    --warmup 0.0         \
    --base_lr 0.007         \
    --CLAHE 0         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 10000         \
    --autodl         \
    --fpn_out_c 256 \
    --sr_out_c 256 \
    --backbone resnet50


python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp crop_IDRID_smp_SR_Unet_ex/resnet50_shuffle/imgz1440_bs2_SGD_CLAHE2_warmup0_newschduler_max_iterations-5000_polyv2_lr1e-2        \
    --dataset_name crop_IDRID         \
    --val_period 54         \
    --image_size 1440         \
    --model SR_Unet         \
    --optim SGD         \
    --batch_size 2         \
    --warmup 0.0         \
    --base_lr 0.01         \
    --CLAHE 2         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 5000         \
    --autodl         \
    --fpn_out_c 256 \
    --backbone resnet50






