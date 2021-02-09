#!/bin/bash

# CUDA DEVICE ID
DEVICE=$1
NUM=$2
DATA=$3
MODEL="slot-attention-clevr-state-$NUM"
DATASET=clevr-state
OUTPATH="out/clevr-state/$MODEL-$ITER"
#-------------------------------------------------------------------------------#
# Train on CLEVR_v1 with cnn model

CUDA_VISIBLE_DEVICES=$DEVICE python train.py --data-dir $DATA --dataset $DATASET --epochs 1500 --ap-log 10 --name $MODEL --lr 0.0004 --batch-size 512 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 
