# pretrained
#6、 SR_Unet_woSR-fpn_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/UNet-from-vessel-pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --ckpt_weight /root/autodl-tmp/Semi_Zoo/exp_2d_vessel/DRIVE/UNet/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4/version01/iter_2000.pth \
    --backbone resnet50 \
    --exclude_keys segmentation_head \
    --autodl



python train_odoc_supervised_2d_smp_aux_pseudo_vessel.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_Unet-PSEUDO-Vessel/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

python train_odoc_supervised_2d_smp_aux_pseudo_vessel.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_SR_Unet_woSR-fpn_pretrained-only-PSEUDO/resnet50/imgz256_bs16-8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 16         \
    --labeled_bs 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --fpn_pretrained \
    --autodl

    --v_cross_c \



#6、 SR_Unet_woSR-fpn_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/SR_Unet_woSR-fpn_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model c         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --fpn_pretrained \
    --backbone resnet50 \
    --autodl

####
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_SR_Unet_woSR-fpn_pretrained/vessel_loss_weight1e/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --fpn_pretrained \
    --vessel_loss_weight 1.0 \
    --autodl


python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_Unet/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --fpn_pretrained \
    --autodl

#myArray+=(item)

python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_Unet/HRF-CHASEDB1-DRIVE_resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --add_vessel HRF-CHASEDB1-DRIVE \
    --autodl


python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp RIM-ONE/Dual_Decoder_Unet/vessel_loss_weight2e-1_addHRF-CHASEDB1_resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE         \
    --image_size 256         \
    --model Dual_Decoder_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --add_vessel HRF-CHASEDB1 \
    --vessel_loss_weight 0.2 \
    --autodl