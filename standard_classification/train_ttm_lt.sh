#!/bin/bash

# 获取输入的单个参数
alpha=$1             # 设定的 alpha 值
beta=$2              # 设定的 beta 值
teacher_model=$3     # Teacher model 
student_model=$4     # Student model 
gpu_id=$5            # GPU ID
b=$6                 # 蒸馏权重
ttm_l=$7             # TTM 专属参数 ttm_l

# ========================================================
# 运行【纯 TTM】(alpha=1, beta=0, ttm_l=0.1): 
# bash train_ttm_lt.sh 1.0 0.0 resnet110 resnet44 0 76 0.1
# ========================================================

echo "=========================================================="
echo "Starting LT_TTM | Alpha: $alpha, Beta: $beta, b: $b, ttm_l: $ttm_l"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=$gpu_id python3 train_student.py \
    --path_t ./save/models/"$teacher_model"_vanilla/ckpt_epoch_240.pth \
    --model_s "$student_model" \
    --dataset cifar100_lt \
    --imb_factor 0.01 \
    --distill ttm \
    --ttm_l ${ttm_l} \
    -r 1 -b "$b" -a 0 \
    --ab_alpha "$alpha" \
    --ab_beta "$beta" \
    --trial LT_TTM_a${alpha}_b${beta}