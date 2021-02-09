#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="slot-attention-clevr-objdiscovery-stacked-$NUM"
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --device-list-parallel $DEVICE --data-dir $DATA --epochs 10 \
--warm-up-steps 500 --test-log 10 --name $MODEL --lr 0.0004 --batch-size 64 --n-slots 11 --n-attr-slots 5 \
--n-iters-slot-att 3
