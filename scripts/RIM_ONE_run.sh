#6、 SR_Unet_woSR-fpn_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp RIM-ONE/SR_Unet_woSR-fpn_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name RIM-ONE        \
    --image_size 256         \
    --model SR_Unet_woSR         \
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