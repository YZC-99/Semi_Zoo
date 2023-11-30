
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/RIM-ONE_Vessel_add \
        --fuse_type add



OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/UNet_MiT_two_Decoder_RIM-ONE_Vessel_add \
        --fuse_type add \
        --model UNet_two_Decoder \
        --backbone b2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/UNet_ResNet_two_Decoder_RIM-ONE_Vessel_add \
        --fuse_type add \
        --model UNet_two_Decoder \
        --backbone resnet50

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d_odoc_vessel.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/RIM-ONE_Vessel_None




########################

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/UNet_MiT_RIM-ONE \
        --model UNet_MiT \
        --backbone b2



OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/UNet_lr-decouple_MiT_RIM-ONE \
        --model UNet_MiT \
        --lr_decouple \
        --backbone b2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp supervised/UNet_ResNet_RIM-ONE \
        --model UNet_ResNet \
        --backbone resnet50

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp supervised/UNet_ResNet_RIM-ONE \
        --model UNet_ResNet \
        --backbone resnet50 \
        --ohem 0.5