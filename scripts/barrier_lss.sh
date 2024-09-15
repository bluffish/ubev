python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline/19.pt --ep_mode entropy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1/19.pt --ep_mode entropy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_ensemble.yaml -e ./outputs_bin/nuscenes/barrier/lss_baseline/19.pt ./outputs_bin/nuscenes/barrier/lss_baseline_seed=1/19.pt ./outputs_bin/nuscenes/barrier/lss_baseline_seed=2/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_ensemble.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1/19.pt ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1_seed=1/19.pt ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1_seed=2/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_dropout.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_dropout.yaml -p ./outputs_bin/nuscenes/barrier/lss_baseline_gamma=1/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_evidential.yaml -p ./outputs_bin/nuscenes/barrier/lss_evidential/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_evidential.yaml -p ./outputs_bin/nuscenes/barrier/lss_evidential_focal/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_energy_ol=.0001/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_baseline.yaml -p ./outputs_bin/nuscenes/barrier/lss_energy_gamma=1_ol=.0001/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_evidential.yaml -p ./outputs_bin/nuscenes/barrier/lss_uce_ol=.01_k=0/18.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_lss_evidential.yaml -p ./outputs_bin/nuscenes/barrier/lss_ufocal_gamma=.5_ol=.01_k=64/18.pt -g 0 1