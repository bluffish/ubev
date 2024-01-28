#!/bin/bash
python train.py cvt baseline nuscenes\
    -g 0 1 --loss ce\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --train_set train_id --val_set val_id -c vehicle