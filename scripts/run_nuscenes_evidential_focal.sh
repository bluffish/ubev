#!/bin/bash
python train.py lss evidential nuscenes\
    -g 6 7 --ood --loss focal\
    --weight_decay 0.0000001 --learning_rate 0.01\
    --gamma 0.1 --beta 0 --ol 0.1 --k 64\
    --train_set train_id_pseudo --val_set val_id_pseudo -c vehicle