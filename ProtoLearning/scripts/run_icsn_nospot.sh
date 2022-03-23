#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ProtoLearning/train_icsn.py \
--save-step 200 --print-step 1 --learning-rate 0.0001 --batch-size 128 --epochs 7200 \
--prototype-vectors 6 6 6 --exp-name icsn-0-ecr_nospot-extramlp \
--seed 0 \
--dataset ecr_nospot --initials WS --lr-scheduler-warmup-steps 1000 \
--data-dir Data/ECR --proto-dim 128 --extra-mlp-dim 1 \
--multiheads --results-dir ProtoLearning/runs/ --temperature 2. \
--n-workers 0 --train-protos