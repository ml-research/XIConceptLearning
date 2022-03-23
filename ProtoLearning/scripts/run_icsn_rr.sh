#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ProtoLearning/train_icsn_rr.py \
--save-step 20 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 200 \
--prototype-vectors 6 6 6 --exp-name icsn-rr-0-ecr --seed 0 \
--dataset ecr --initials WS --lr-scheduler-warmup-steps 1000  \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 0  \
--multiheads --results-dir ProtoLearning/runs/ --temperature 2. \
--ckpt-fp ProtoLearning/runs/icsn-0-ecr/states/07999.pth \
--wrong-protos 2 3 --wrong-protos 2 4 --wrong-protos 0 2 3 5 \
--n-workers 0 --train-protos --lambda-rr 10