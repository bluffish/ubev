#!/bin/bash
python train.py cvt baseline nuscenes\
    -g 2 3 --loss focal\
    --gamma 0.1\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --train_set train_id --val_set val_id -c vehicle