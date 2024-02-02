#!/bin/bash
echo  "python train.py ./configs/train_nuscenes_lss_evidential.yaml -g 0 1 -l ./outputs_bin/nuscenes/vehicle/lss_ufocal_gamma=.05 --loss focal --gamma .05"
sleep 3600
echo "RUNNING NOW"
python train.py ./configs/train_nuscenes_lss_evidential.yaml -g 6 7 -l ./outputs_bin/nuscenes/vehicle/lss_ufocal_gamma=.05 --loss focal --gamma .05
