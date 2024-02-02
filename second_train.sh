#!/bin/bash
echo  "python train.py ./configs/train_nuscenes_lss_evidential.yaml --ol .01 --k 0 -g 2 3 -o -l ./outputs_bin/nuscenes/aug/lss_uce_ol=.01_k=0"
sleep 1800
echo "RUNNING NOW"
python python train.py ./configs/train_nuscenes_lss_evidential.yaml --ol .01 --k 0 -g 2 3 -o -l ./outputs_bin/nuscenes/aug/lss_uce_ol=.01_k=0

