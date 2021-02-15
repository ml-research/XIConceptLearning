#!/bin/bash

# CUDA DEVICE ID
DEVICE=0
NUM=$1
MODEL="convae-toydata-protomixture-v1-a-$NUM"
#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python unsup_prototype_group/train_v1_a.py --data-dir data --epochs 1500 \
--warm-up-steps 0 --test-log 50 --name $MODEL --lr 1e-3 --batch-size 1024 \
--lambda-recon-imgs 0. --lambda-recon-protos 5. --lambda-r1 1e-2 --lambda-r2 0. --lambda-ad .0 --lambda-enc-mse 0.0 \
--img-shape 3,28,28 --n-prototype-groups 1 --n-prototype-vectors-per-group 6 --seed 1111