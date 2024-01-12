#1、 UNet
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/UNet/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model UNet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --autodl
#2、 SR_Unet_woFPN
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet_woFPN/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet_woFPN         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --autodl

#3、 SR_Unet_woFPN-sr_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet_woFPN-sr_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet_woFPN         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --sr_pretrained \
    --autodl


#4、 SR_Unet_woSR
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet_woSR/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --autodl

#5、 SR_Unet_woSR-fpn_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet_woSR-fpn_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --fpn_pretrained \
    --backbone resnet50 \
    --autodl

#6、 SR_Unet
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --autodl

#7、 SR_Unet-fpn_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet-fpn_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --fpn_pretrained \
    --autodl

#8、 SR_Unet-sr_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet-sr_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --sr_pretrained \
    --autodl


#9、 SR_Unet-fpn_pretrained-sr_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet-fpn_pretrained-sr_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --sr_pretrained \
    --fpn_pretrained \
    --autodl


#10、Dual_Decoder_SR_Unet_woSR
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet_woSR/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woSR         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

#11、Dual_Decoder_SR_Unet_woSR-fpn_pretrained
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet_woSR-fpn_pretrained/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
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
    --autodl


#12、Dual_Decoder_SR_Unet_woFPN
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet_woFPN/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woFPN         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

#13、Dual_Decoder_SR_Unet_woFPN-sr_pretrained
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet_woFPN-sr_pretrained/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet_woFPN         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --sr_pretrained \
    --backbone resnet50 \
    --autodl

# 14、Dual_Decoder_SR_Unet
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --backbone resnet50 \
    --autodl

# 15、Dual_Decoder_SR_Unet-fpn_pretrained
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet-fpn_pretrained/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --fpn_pretrained \
    --backbone resnet50 \
    --autodl

# 16、Dual_Decoder_SR_Unet-sr_pretrained
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet-sr_pretrained/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --sr_pretrained \
    --backbone resnet50 \
    --autodl

# 17、Dual_Decoder_SR_Unet-fpn_pretrained-sr_pretrained
python train_odoc_supervised_2d_smp_aux.py \
    --num_works 2        \
    --device 0         \
    --exp REFUGE/Dual_Decoder_SR_Unet-fpn_pretrained-sr_pretrained/resnet50/imgz256_bs8-4_Adam_warmup0_iterations-3000_polyv2_lr3e-4        \
    --dataset_name REFUGE         \
    --image_size 256         \
    --model Dual_Decoder_SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --labeled_bs 4         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 3000         \
    --sr_pretrained \
    --fpn_pretrained \
    --backbone resnet50 \
    --autodl

###TODO
# 18、Dual_Decoder_SR_Unet-fpn_pretrained-sr_pretrained
python train_odoc_supervised_2d_smp.py \
    --num_works 4        \
    --device 0         \
    --exp REFUGE/SR_Unet-fpn_pretrained-sr_pretrained/resnet50/imgz256_bs8_Adam_warmup0_iterations-6400_polyv2_lr3e-4        \
    --dataset_name REFUGE        \
    --image_size 256         \
    --model SR_Unet         \
    --optim Adam         \
    --batch_size 8         \
    --warmup 0.0         \
    --base_lr 0.0003         \
    --max_iterations 6400         \
    --backbone resnet50 \
    --sr_pretrained \
    --fpn_pretrained \
    --autodl
