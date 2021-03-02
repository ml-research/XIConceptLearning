#!/bin/bash

# sum
python prototype_slots_group_var2/train_agglayer.py --device cuda --lambda-pair 1. --lambda-enc-mse 0. \
--lambda-recon-proto 0. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 20 \
--learn weakly --softmax-temp .1 \
--exp-name test_attrpred --agg-type linear