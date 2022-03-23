#!/bin/bash

DEVICE=$1

CUDA_VISIBLE_DEVICES=0 python ProtoLearning/train_icsn_novelshape.py \
--save-step 200 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 600 \
--prototype-vectors 6 6 6 --exp-name icsn-0-ecr-novelshape \
--seed 0 \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 1000 \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 0 \
--multiheads --results-dir ProtoLearning/runs/ --temperature 2. \
--n-workers 0 --train-protos --ckpt-fp WeakAEProtoLearning/runs/icsn-rr2-0-ecr/states/00799.pth