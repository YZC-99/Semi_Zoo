OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup python train_idrid_supervised_2d.py \
        --num_works 0 \
        --device 0 \
        --exp IDRID_512 \
        --dataset_name IDRID \
        --val_period 15 \
        --image_size 512 \
        --model UNet_efficient \
        --batch_size 4 \
        --base_lr 0.0003 \
        --CLAHE \
        --ce_weight 0.001 0.1 0.1 0.1 0.1 \
        --backbone b0 &


        0.001,0.1,0.1,0.1,0.1
        1,2,2,2,2