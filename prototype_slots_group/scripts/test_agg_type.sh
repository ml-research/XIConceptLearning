#!/bin/bash

# sum
python prototype_slots_group/train_agglayer.py --device cuda --lambda-pair 0. --lambda-enc-mse 0. \
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 20 \
--learn weakly --temp .1 \
--exp-name test_agg_sum --agg-type sum

# linear
python prototype_slots_group/train_agglayer.py --device cuda --lambda-pair 0. --lambda-enc-mse 0. \
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 20 \
--learn weakly --temp .1 \
--exp-name test_agg_linear --agg-type linear

# simple_attention
python prototype_slots_group/train_agglayer.py --device cuda --lambda-pair 0. --lambda-enc-mse 0. \
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 20 \
--learn weakly --temp .1 \
--exp-name test_agg_simple_attention --agg-type simple_attention

# attention
python prototype_slots_group/train_agglayer.py --device cuda --lambda-pair 0. --lambda-enc-mse 0. \
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 0. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 20 \
--learn weakly --temp .1 \
--exp-name test_agg_attention --agg-type attention
