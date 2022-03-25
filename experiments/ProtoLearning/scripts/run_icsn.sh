#!/bin/bash

DEVICE=$1
SEED=$2

CUDA_VISIBLE_DEVICES=$DEVICE python experiments/ProtoLearning/train_icsn.py \
--save-step 200 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 8000 \
--prototype-vectors 6 6 6 --exp-name icsn-$SEED-ecr \
--seed $SEED \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 1000 \
--data-dir data/ECR --proto-dim 128 --extra-mlp-dim 0 \
--multiheads --results-dir experiments/ProtoLearning/runs/ --temperature 2. \
--n-workers 0 --train-protos
