#!/bin/bash

python prototype_slots_group_var2/train.py --device cuda --lambda-pair 1. --lambda-enc-mse 0. \
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 2 \
--learn weakly --softmax-temp .1 \
--exp-name train --agg-type linear