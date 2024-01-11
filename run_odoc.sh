python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE-SR_Unet_woSR/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr1e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0001         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

    SR_Unet_woSR



# python train_odoc_supervised_2d_smp_aux.py \
#     --num_works 2       \
#     --device 0         \
#     --exp RIM-ONE-Dual_Decoder_Unet-aux/resnet50/vessel_loss_weight1e_imgz256_bs16_Adam_warmup0_iterations-3000_polyv2_lr2e-4        \
#     --dataset_name RIM-ONE         \
#     --image_size 256         \
#     --model Dual_Decoder_Unet         \
#     --optim Adam         \
#     --batch_size 16         \
#     --labeled_bs 8         \
#     --warmup 0.0         \
#     --base_lr 0.0002         \
#     --max_iterations 3000         \
#     --backbone resnet50 \
#     --vessel_loss_weight 1.0 \
#     --autodl



python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE-Dual_Decoder_Unet-aux/resnet50/vessel_loss_weight1e_imgz256_bs16_SGD_warmup0_iterations-3000_polyv2_lr1e-2        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim SGD         \
    --batch_size 16         \
    --labeled_bs 8         \
    --warmup 0.0         \
    --base_lr 0.01         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --vessel_loss_weight 1.0 \
    --autodl

    --fpn_pretrained \
    --sr_pretrained \
# fpn_pretrained
# sr_pretrained
Dual_Decoder_SR_Unet

