#!/bin/bash

# Configuration
python_path="/home/yanhongwei/miniconda3/envs/DGIL/bin/python"
data_dir="/data/datasets/cifar100"
seeds=(0 1 2)
gpus=(5 6 7)

# Run experiments
for i in {0..2}; do
    seed=${seeds[$i]}
    gpu=${gpus[$i]}
    exp_name="r32_space_1.5_seed${seed}"

    echo "Starting $exp_name on GPU $gpu"

    screen -dmS "$exp_name" bash -c "
        cd $(pwd) && \
        CUDA_VISIBLE_DEVICES=$gpu \
        $python_path -W ignore main.py \
            --save_name $exp_name \
            --model_num 2 \
            --backbone resnet32 \
            --data_dir $data_dir \
            --space_interval 1.5 \
            --epochs 400 \
            --random_seed $seed
    "
done

echo "All experiments started. Use 'screen -ls' to list sessions."