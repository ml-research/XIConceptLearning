#!/bin/bash

DEVICE=$1
SEED=$2

CUDA_VISIBLE_DEVICES=$DEVICE python icsn/train_introae.py \
--save-step 50 --print-step 1 --learning-rate 0.00004 --batch-size 256 --epochs 500 \
--exp-name introae-$SEED-ecr --seed $SEED \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 400 \
--data-dir data/ECR --results-dir icsn/runs/