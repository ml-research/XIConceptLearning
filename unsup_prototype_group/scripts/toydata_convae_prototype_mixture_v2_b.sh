#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
MODEL="convae-toydata-protomixture-v2-b-$NUM"
#-------------------------------------------------------------------------------#

CUDA_VISIBLE_DEVICES=$DEVICE python train_v2_a.py --data-dir data --epochs 100 \
--warm-up-steps 200 --test-log 1 --name $MODEL --lr 0.005 --batch-size 100 \
--lambda-recon-imgs 1. --lambda-recon-protos 1. --lambda-r1 1. --lambda-r2 1. --lambda-ad 1. \
--img-shape 3,28,28 --n-prototype-groups 2 --n-prototype-vectors-per-group 4