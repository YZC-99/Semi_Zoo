python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp crop_IDRID_smp_LightNet_wFPN-pre_wSR-pre_ex/resnet50_shuffle/imgz1440_bs2_Adam_CLAHE2_warmup0_newschduler_max_iterations-5000_polyv2_lr3e-4         \
    --dataset_name crop_IDRID         \
    --val_period 54         \
    --image_size 1440         \
    --model LightNet_wFPN_wSR         \
    --optim Adam         \
    --batch_size 2         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --CLAHE 2         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 5000         \
    --autodl         \
    --fpn_out_c 256 \
    --sr_out_c 256 \
    --sr_pretrained \
    --fpn_pretrained \
    --backbone resnet50


python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp crop_IDRID_smp_SR_Unet_ex/resnet50_shuffle/imgz1024_bs2_Adam_CLAHE2_warmup0_newschduler_max_iterations-5000_polyv2_lr3e-4         \
    --dataset_name crop_IDRID         \
    --val_period 54         \
    --image_size 1024         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 2         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --CLAHE 2         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 5000         \
    --autodl         \
    --fpn_out_c 256 \
    --backbone resnet50






