OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE_vessel \
        --dataset_name RIM-ONE \
        --fuse_type None \
        --model UNet_two_Decoder \
        --backbone org


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/RIM-ONE_vessel \
        --dataset_name RIM-ONE \
        --fuse_type ccrtb3 \
        --model UNet_two_Decoder \
        --backbone resnet50 \
        --batch_size 8 \
        --labeled_bs 4 \
        --base_lr 0.0001 \
        --max_iterations 10000 \
        --ce_weight

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE_vessel \
        --dataset_name RIM-ONE \
        --fuse_type ccrtb3 \
        --model UNet_two_Decoder \
        --backbone resnet50 \
        --batch_size 8 \
        --labeled_bs 4 \
        --base_lr 0.0001 \
        --max_iterations 10000 \
        --ohem 0.5


#
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/REFUGE_vessel \
        --dataset_name REFUGE \
        --fuse_type ccrtb3 \
        --model UNet_two_Decoder \
        --labeled_num 400 \
        --total_num 445 \
        --backbone resnet50 \
        --batch_size 8 \
        --labeled_bs 4 \
        --base_lr 0.0001 \
        --max_iterations 10000 \
        --ce_weight








########################

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE \
        --model UNet


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE \
        --model UNet_MiT \
        --lr_decouple \
        --backbone b2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE \
        --model UNet_MiT \
        --backbone b4

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/RIM-ONE \
        --model UNet_ResNet \
        --backbone resnet50 \
        --ohem 0.5

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --dataset_name REFUGE \
        --exp supervised/REFUGE \
        --model UNet_ResNet \
        --backbone resnet34 \
        --base_lr 0.0001 \
        --batch_size 8


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE \
        --model Segformer \
        --backbone b2 \
        --batch_size 8 \
        --base_lr 0.0001



OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/RIM-ONE \
        --model Deeplabv3+ \
        --backbone resnet50 \
        --batch_size 8 \
        --ohem 0.5 \
        --base_lr 0.0001





#############################IDRID#########################
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_dr_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --val_period 10 \
        --model UNet

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_dr_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --val_period 20 \
        --model SR_UNet_ResNet \
        --batch_size 4 \
        --base_lr 0.00005 \
        --backbone resnet34

        UNet_ResNet Deeplabv3p SR_UNet_ResNet
