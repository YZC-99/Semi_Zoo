OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d.py \
        --num_works 0 \
        --device 1 \
        --exp IDRID_1024_weight_2 \
        --dataset_name IDRID \
        --val_period 15 \
        --image_size 1440 \
        --model UNet_efficient \
        --batch_size 1 \
        --base_lr 0.0001 \
        --CLAHE 4\
        --ce_weight 0.001 1.0 1.0 1.0 1.0 \
        --backbone b0 &


        0.001,0.1,0.1,0.1,0.1
        1,2,2,2,2