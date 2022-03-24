#!/bin/bash

DEVICE=$1

CUDA_VISIBLE_DEVICES=0 python ProtoLearning/train_icsn_spot.py \
--save-step 50 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 2000 \
--prototype-vectors 6 6 6 6 --exp-name icsn-0-ecr_spot --seed 0 \
--dataset ecr_spot --initials WS --lr-scheduler-warmup-steps 1000  \
--data-dir data/ECR --proto-dim 128 --extra-mlp-dim 0  \
--multiheads --results-dir experiments/ProtoLearning/runs/ \
--ckpt-fp experiments/ProtoLearning/runs/icsn-rr-0-ecr_spot-extramlp/states/00999.pth \
--n-workers 0 --lambda-rr 1 --train-protos
