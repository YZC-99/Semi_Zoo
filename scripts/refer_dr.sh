
python test_idrid_supervised_2d_smp.py \
        --num_works 8 \
        --device 1 \
        --exp crop_IDRID/Unet_wFPN_wSKA/se_resnet50 \
        --dataset_name crop_IDRID \
        --image_size 1024 \
        --model Unet_wFPN_wSKA \
        --batch_size 2 \
        --base_lr 0.0008 \
        --CLAHE 0 \
        --autodl \
        --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_dr/crop_IDRID/Unet_wFPN_wSKA/lr8e-4_poly-v2-interval-30/imgz1024_se_resnet50/bs2_Adam_CLAHE0-5k/version02/best_AUC_PR_EX0.8143_iter_4266.pth \
        --backbone se_resnet50

