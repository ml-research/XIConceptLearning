#!/bin/bash

DEVICE=$1
SEED=$2

CUDA_VISIBLE_DEVICES=$DEVICE python icsn/train_icsn.py \
--save-step 50 --print-step 1 --learning-rate 0.00004 --batch-size 256 --epochs 500 \
--prototype-vectors 6 6 6 --exp-name icsn-$SEED-ecr-extradim1 --train-protos \
--seed $SEED \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 400 \
--data-dir data/ECR --results-dir icsn/runs/ --proto-dim 128 --extra-mlp-dim 1 \
--multiheads --temperature 2. --freeze-enc \
--n-workers 0 --train-protos \
--temp-scheduler-step 100 --temp-scheduler-rate 0.2 \
--pretrained-model icsn/runs/introae-$SEED-ecr/states/model.pth