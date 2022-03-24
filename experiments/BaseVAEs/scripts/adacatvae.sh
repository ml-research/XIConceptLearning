#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python BaseVAEs/train_disent_adacatvae.py --save-step 200 --print-step 1 --learning-rate 0.0001 \
--batch-size 128 --epochs 2000 --exp-name adaid-catvae-21-ecr \
--n-groups 3 --n-protos 6 --seed 21 --dataset ecr --initials WS \
--lr-scheduler-warmup-steps 1000 --data-dir data/ECR/ \
--results-dir experiments/BaseVAEs/runs/ --n-workers 0 \
--temperature 0.1
