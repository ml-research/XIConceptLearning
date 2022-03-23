#!/bin/bash

DEVICE=$1

CUDA_VISIBLE_DEVICES=0 python ProtoLearning/train_icsn_nospot_rr.py \
--save-step 50 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 1000 \
--prototype-vectors 6 6 6 --exp-name icsn-rr-0-ecr_spot-extramlp --seed 0 \
--dataset ecr_spot --initials WS --lr-scheduler-warmup-steps 1000  \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 1  \
--multiheads --results-dir ProtoLearning/runs/ \
--ckpt-fp ProtoLearning/runs/icsn-0-ecr_nospot-extramlp/states/07999.pth \
--n-workers 0 --train-protos --lambda-rr 10