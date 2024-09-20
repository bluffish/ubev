#!/bin/bash
python train.py lss baseline nuscenes\
    -g 4 5 --loss focal\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --train_set train_id_pseudo --val_set val_id_pseudo -c vehicle