#!/bin/bash
python train.py cvt evidential nuscenes\
    -g 4 5 --loss ce\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --gamma 0 --beta 0.0001 --ol 0 --k 0\
    --train_set train_id --val_set val_id -c vehicle