
#
# smp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d_smp.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_smp_UNet \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model DeepLabV3p \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --backbone resnet34 &


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d_smp_BCE.py \
        --num_works 0 \
        --device 0 \
        --exp IDRID_smp_UNet_binary_mean_std_ratio_weight \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --backbone resnet34


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d_smp_BCE.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_smp_UNet_binary_mean_std_100weight \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --backbone resnet34

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d_smp_BCE.py \
        --num_works 0 \
        --device 0 \
        --exp IDRID_smp_UNet_binary_noback_mean_std\
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --batch_size 1 \
        --base_lr 0.00002 \
        --CLAHE 4 \
        --backbone resnet34

# no background
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d_smp_BCE_NOBack.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_smp_UNet_binary_noback \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --backbone resnet34