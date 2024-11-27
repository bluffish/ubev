#!/bin/bash

# Read command-line arguments
GPU_IDS=$1
NUSCENES_C=$2

# Validate arguments
if [ -z "$GPU_IDS" ] || [ -z "$NUSCENES_C" ]; then
  echo "Usage: $0 <gpu_ids> <nuscenes_c_value>"
  exit 1
fi

# Run the commands with the arguments
python eval_nusc.py ./configs/eval_nuscenes_cvt_baseline.yaml \
  -p ./outputs/nuscenes/cvt_baseline_gamma=.05_ol=.0001/19.pt \
  --ep_mode energy -g $GPU_IDS --nuscenes_c $NUSCENES_C

python eval_nusc.py ./configs/eval_nuscenes_cvt_evidential.yaml \
  -p ./outputs/nuscenes/cvt_evidential_ol=.01/19.pt \
  -g $GPU_IDS --nuscenes_c $NUSCENES_C

python eval_nusc.py ./configs/eval_nuscenes_cvt_evidential.yaml \
  -p ./outputs/nuscenes/cvt_evidential_gamma=.05_ol=.01_k=64/19.pt \
  -g $GPU_IDS --nuscenes_c $NUSCENES_C
