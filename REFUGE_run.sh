#
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_REFUGE_contours.py \
        --num_works 0 \
        --device 1 \
        --dataset_name REFUGE \
        --exp supervised/REFUGE_contour \
        --model UNet_two_Decoder \
        --backbone resnet34 \
        --base_lr 0.0003 \
        --batch_size 4


###

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_REFUGE.py \
        --num_works 0 \
        --device 1 \
        --dataset_name REFUGE \
        --exp supervised/REFUGE \
        --model UNet_ResNet \
        --backbone resnet34 \
        --base_lr 0.0003 \
        --batch_size 4


        --with_ce \
        --with_dice


OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_REFUGE.py \
        --num_works 0 \
        --device 0 \
        --dataset_name REFUGE/train_mix_val_as_train \
        --exp test_demo\
        --model UNet_ResNet \
        --backbone resnet34 \
        --base_lr 0.0001 \
        --batch_size 8



# vessel
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/REFUGE_vessel \
        --dataset_name REFUGE \
        --fuse_type rtb4 \
        --model UNet_two_Decoder \
        --labeled_num 400 \
        --total_num 445 \
        --backbone resnet34 \
        --batch_size 4 \
        --labeled_bs 2 \
        --base_lr 0.0003

        --with_dice


        --with_ce \
        --with_dice




        --max_iterations 10000


# dual encoder
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_dual_encoder.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/REFUGE_dual_encoder \
        --dataset_name REFUGE \
        --model UNet_dual_Encoder \
        --labeled_num 400 \
        --total_num 445 \
        --backbone resnet34 \
        --batch_size 4 \
        --labeled_bs 2 \
        --base_lr 0.0003


# vessel_tri
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel_tri.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/REFUGE_vessel_tri \
        --dataset_name REFUGE \
        --model UNet_tri_Decoder \
        --labeled_num 400 \
        --total_num 445 \
        --backbone resnet34 \
        --batch_size 4 \
        --labeled_bs 2 \
        --base_lr 0.0003