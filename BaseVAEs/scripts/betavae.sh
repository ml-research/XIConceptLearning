#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python BaseVAEs/train_disent_betavae.py --save-step 200 --print-step 1 --learning-rate 0.0001 \
 --batch-size 128 --epochs 2000 --exp-name unsup-betavae-0-ecr \
--n-groups 3 --n-protos 6 --seed 0 --dataset ecr --initials WS \
--lr-scheduler-warmup-steps 1000 --data-dir Data/ECR \
--results-dir BaseVAEs/runs/ --n-workers 0