#!/bin/bash
echo  "python train.py ./configs/train_carla_lss_baseline.yaml -o -g 0 1 -l ./outputs_bin/energy_test"
sleep 2400
echo "RUNNING NOW"
python train.py ./configs/train_carla_lss_baseline.yaml -o -g 0 1 -l ./outputs_bin/energy_test
