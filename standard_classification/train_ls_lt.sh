#!/bin/bash

# 获取输入的单个参数
alpha=$1             # 设定的 alpha 值
beta=$2              # 设定的 beta 值
teacher_model=$3     # Teacher model 
student_model=$4     # Student model
gpu_id=$5            # GPU ID

# LS 特定的硬编码参数
b=9

# ========================================================
# 运行【纯 LS】(alpha=1, beta=0): 
# bash train_ls_lt.sh 1.0 0.0 resnet110 resnet44 0
# ========================================================

echo "=========================================================="
echo "Starting LT_LS | Alpha: $alpha, Beta: $beta, b: $b"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=$gpu_id python3 train_student.py \
    --path_t ./save/models/"$teacher_model"_vanilla/ckpt_epoch_240.pth \
    --model_s "$student_model" \
    --dataset cifar100_lt \
    --imb_factor 0.01 \
    --distill ls \
    -r 0.1 -b "$b" -a 0 \
    --kd_T 2 \
    --ab_alpha "$alpha" \
    --ab_beta "$beta" \
    --trial LT_LS_a${alpha}_b${beta}