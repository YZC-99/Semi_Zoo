


#1、 UNet
python train_supervised_2d_odoc_vessel.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_SR_Unet_woFPN_vessel-loss-weight-1e-1/resnet50_only-pseudo-cover-oc-rim-50p/imgz256_bs8_AdamW_warmup0_iterations-4000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woFPN         \
    --optim AdamW         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 4000         \
    --backbone resnet50 \
    --vessel_type oc-rim50 \
    --vessel_loss_weight 0.1 \
    --autodl

minus-od-rim100
sr_pretrained
fpn_pretrained
Dual_Decoder_Unet_only-pseudo-cover-oc-rim-10p
