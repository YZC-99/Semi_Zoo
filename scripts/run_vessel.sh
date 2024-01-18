python train_supervised_2d_odoc_vessel.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet_woSR/kink-loss/resnet50_only-pseudo-cover-oc-rim-50p  \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woSR        \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 5000         \
    --backbone resnet50 \
    --vessel_type oc-rim50 \
    --KinkLoss 1.0 \
    --autodl

center-detach
    --fpn_pretrained \







python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp CLAHE4-8-REFUGE/UNet/resnet50  \
    --dataset_name CLAHE4-8-REFUGE        \
    --image_size 256         \
    --model UNet        \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 5000         \
    --backbone resnet50 \
    --autodl


    --decoder_attention_type scse \

SR_Unet_woSR-fpn_pretrained
        --fpn_pretrained \
        UNet





#1、 UNet
python train_supervised_2d_odoc_vessel.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_SR_Unet_vessel-sr_pretrained-fpn_pretrained/resnet50_only-pseudo-cover-oc-rim-50p/imgz256_bs8_AdamW_warmup0_iterations-4000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet         \
    --optim AdamW         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 4000         \
    --backbone resnet50 \
    --vessel_type oc-rim50 \
    --vessel_loss_weight 0.1 \
    --sr_pretrained \
    --fpn_pretrained \
    --autodl

minus-od-rim100
sr_pretrained
fpn_pretrained
Dual_Decoder_Unet_only-pseudo-cover-oc-rim-10p
