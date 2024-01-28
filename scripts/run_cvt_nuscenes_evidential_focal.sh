#!/bin/bash
python train.py cvt evidential nuscenes\
    -g 6 7 --loss focal\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --gamma 0.1 --beta 0 --ol 0 --k 0\
    --train_set train_id --val_set val_id -c vehicle