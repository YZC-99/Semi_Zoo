python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp crop_IDRID_smp_SR_Unet-SR-pre_ex/resnet50_shuffle/imgz1024_bs4_Adam_CLAHE0_warmup0_newschduler_max_iterations-4000_polyv2_lr3e-4        \
    --dataset_name crop_IDRID         \
    --val_period 54         \
    --image_size 1024         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --CLAHE 0         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 4000         \
    --autodl         \
    --fpn_out_c 256 \
    --sr_out_c 256 \
    --backbone resnet50 \
    --sr_pretrained


    \
    --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/patch_IDRID_smp_UNet_ex/resnet50_shuffle/imgz1024_bs4_SGD_CLAHE0_warmup0_newschduler_max_iterations-10000_polyv2_lr7e-3/version0/best_AUC_PR_EX0.734_iter_4860.pth


python train_idrid_supervised_2d_smp.py \
    --num_works 8        \
    --device 0         \
    --exp crop_IDRID-from-patch-pretrained_smp_LightNet_wFPN_wSR_ex/resnet50_shuffle/imgz1024_bs4_Adam_CLAHE0_warmup0_newschduler_max_iterations-10000_polyv2_lr7e-3        \
    --dataset_name crop_IDRID         \
    --val_period 54         \
    --image_size 1024         \
    --model LightNet_wFPN_wSR         \
    --optim Adam         \
    --batch_size 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --CLAHE 0         \
    --ce_weight 1 1 1 1 1         \
    --max_iterations 5000         \
    --autodl         \
    --fpn_out_c 256 \
    --sr_out_c 256 \
    --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/patch_IDRID_smp_LightNet_wFPN_wSR_ex/resnet50_shuffle/imgz1024_bs4_Adam_CLAHE0_warmup0_newschduler_max_iterations-5000_polyv2_lr3e-4/version01/best_AUC_PR_EX0.752_iter_2997.pth \
    --backbone resnet50



