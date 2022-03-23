#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python ProtoLearning/train_icsn_rr_iter2.py \
--save-step 50 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 200 \
--prototype-vectors 6 6 6 --exp-name icsn-rr2-0-ecr --seed 0 \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 1000  \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 0  \
--multiheads --results-dir ProtoLearning/runs/ \
--ckpt-fp ProtoLearning/runs/icsn-rr-0-ecr/states/00199.pth \
--wrong-protos 2 3 --wrong-protos 2 4 --wrong-protos 0 2 3 5 \
--n-workers 0 --train-protos --lambda-rr 1

CUDA_VISIBLE_DEVICES=8 python ProtoLearning/train_icsn_rr_iter2.py \
--save-step 50 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 200 \
--prototype-vectors 6 6 6 --exp-name icsn-rr2-13-ecr --seed 13 \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 1000  \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 0  \
--multiheads --results-dir ProtoLearning/runs/ \
--ckpt-fp ProtoLearning/runs/icsn-rr-13-ecr/states/00199.pth \
--wrong-protos 0 3 --wrong-protos 2 3 --wrong-protos 0 1 2 3 \
--n-workers 0 --train-protos --lambda-rr 1