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

# aux
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d_aux.py \
        --num_works 0 \
        --device 0 \
        --exp IDRID_AUX \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet_efficient_aux \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE \
        --backbone b0 \
        --aux_weight 0.2 &

# no-aux
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --val_period 10 \
        --model UNet




#
# smp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d_smp.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_smp_UNet_ex \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model DeepLabV3p \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --backbone resnet34 &


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_idrid_supervised_2d_smp.py \
        --num_works 0 \
        --device 0 \
        --exp IDRID_smp_UNet_ex_skmetric \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4 \
        --ce_weight 0.001 1.0 0.1 0.1 0.1 \
        --backbone resnet34


        --ce_weight 0.001 1.0 1.0 1.0 1.0 \
###
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_weight1 \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet_efficient \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 3 \
        --backbone b0 &


        --ce_weight 0.001 0.1 0.1 0.1 0.1 \
        --annealing_softmax_focalloss UNet_MiT

UNet_ResNet Deeplabv3p SR_UNet_ResNet UNet_efficient UNet_MiT UNet_efficient_SR

        --ce_weight 0.001 1.0 0.1 0.1 0.1

        --ce_weight 1.0 1.0 1.0 1.0 1.0

        --ce_weight 1.0 2.0 2.0 2.0 2.0
        --ce_weight 0.013692409236879956 24.390243902439025 5.747126436781609 6.493506493506494 27.027027027027028
        --ce_weight 1.0 1.0 1.0 1.0 1.0
        \
        --annealing_softmax_focalloss
        --CLAHE \


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup  python train_idrid_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID \
        --dataset_name IDRID \
        --val_period 54 \
        --image_size 1440 \
        --model UNet_ResNet \
        --batch_size 1 \
        --base_lr 0.0001 \
        --ohem 0.5 \
        --backbone resnet34  &


# DDR
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_ddr_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp DDR \
        --dataset_name DDR \
        --val_period 90 \
        --model UNet_ResNet \
        --batch_size 4 \
        --base_lr 0.00005 \
        --backbone resnet34

        UNet_ResNet Deeplabv3p SR_UNet_ResNet UNet_efficient UNet_MiT
        --dataset_name DDR
        --exp DDR