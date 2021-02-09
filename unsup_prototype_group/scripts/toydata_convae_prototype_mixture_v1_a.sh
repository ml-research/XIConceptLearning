#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
MODEL="convae-toydata-protomixture-v1-a-$NUM"
#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python train_v1_a.py --data-dir data --epochs 100 \
--warm-up-steps 200 --test-log 1 --name $MODEL --lr 0.001 --batch-size 128 \
--lambda-recon-imgs 0. --lambda-recon-protos 5. --lambda-r1 1. --lambda-r2 0. --lambda-ad 1. \
--img-shape 3,28,28 --n-prototype-groups 2 --n-prototype-vectors-per-group 4 --seed 1