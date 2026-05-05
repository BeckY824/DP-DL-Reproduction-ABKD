#!/bin/bash

# 获取输入的单个参数 (剥离了循环)
alpha=$1             # 设定的 alpha 值
beta=$2              # 设定的 beta 值
teacher_model=$3     # Teacher model (e.g., resnet110)
student_model=$4     # Student model (e.g., resnet44)
gpu_id=$5            # GPU ID
dkd_beta=$6          # DKD 专属的 beta 参数 (原版脚本里的 $8)

# ========================================================
# 运行【纯 DKD】(alpha=1, beta=0): 
# bash train_dkd_lt.sh 1.0 0.0 resnet110 resnet44 0 6
# 运行【AB-DKD】(例如 alpha=0.5, beta=1.2): 
# bash train_dkd_lt.sh 0.5 1.2 resnet110 resnet44 0 6
# ========================================================

echo "=========================================================="
echo "Starting LT_DKD | Alpha: $alpha, Beta: $beta, DKD_Beta: $dkd_beta"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=$gpu_id python3 train_student.py \
    --path_t ./save/models/"$teacher_model"_vanilla/ckpt_epoch_240.pth \
    --model_s "$student_model" \
    --dataset cifar100_lt \
    --imb_factor 0.01 \
    --distill dkd \
    -r 1 -b 1 -a 0 \
    --dkd_beta ${dkd_beta} \
    --ab_alpha "$alpha" \
    --ab_beta "$beta" \
    --trial LT_DKD_a${alpha}_b${beta}