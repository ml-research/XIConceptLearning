#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python BaseVAEs/train_disent_adavae_orig.py --save-step 200 --print-step 1 --learning-rate 0.0001 \
--batch-size 128 --epochs 2000 --exp-name adaid-orig-vae-0-ecr-beta4 \
--n-groups 3 --seed 0 --dataset ecr --initials WS \
--lr-scheduler-warmup-steps 1000 --data-dir Data/ECR --proto-dim 128 \
--results-dir BaseVAEs/runs/ --n-workers 0 --beta 4.