#!/bin/bash

python prototype_slots_group/train.py --device cuda --lambda-pair 0. --lambda-enc-mse 0. \
<<<<<<< HEAD
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 1. -lr 0.001 -bs 512 -e 200 -pv 2 4 \
--data-dir Data --seed 13 --dataset toysamecolordifshapepairs --initials WS --print-step 2 \
--learn weakly --temp 1. \
--exp-name train --agg-type linear
=======
--lambda-recon-proto 1. --lambda-r1 0. --lambda-recon-z 1. -lr 0.0001 -bs 512 -e 1000 -pv 4 4 \
--data-dir Data --seed 13 --dataset toycolorshapepairs --initials WS --print-step 10 \
--learn weakly --temp 1. \
--exp-name train_colorshapepairs_sum --agg-type sum
>>>>>>> 36fb5454586e6018eef404a4c6db52ff9c823677
