#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="slot-attention-clevr-objdiscovery-$NUM"
OUTPATH="out/clevr-state/$MODEL-$ITER"
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --device-list-parallel $DEVICE --data-dir $DATA --epochs 500 --warm-up-steps 10000 --test-log 10 --name $MODEL --lr 0.0004 --batch-size 64 --n-slots 11 --n-iters-slot-att 3
#CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --epochs 1 --warm-up-steps 10 --test-log 10 --name $MODEL --lr 0.0004 --batch-size 64 --n-slots 11 --n-iters-slot-att 3
#CUDA_VISIBLE_DEVICES=$DEVICE kernprof -l train.py --data-dir $DATA --epochs 1 --warm-up-steps 10 --test-log 10 --name $MODEL --lr 0.0004 --batch-size 64 --n-slots 11 --n-iters-slot-att 3
