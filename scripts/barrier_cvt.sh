python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline/19.pt --ep_mode entropy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05/19.pt --ep_mode entropy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_ensemble.yaml -e ./outputs_bin/nuscenes/barrier/cvt_baseline/19.pt ./outputs_bin/nuscenes/barrier/cvt_baseline_seed=1/19.pt ./outputs_bin/nuscenes/barrier/cvt_baseline_seed=2/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_ensemble.yaml -e ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05/19.pt ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05_seed=1/19.pt ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05_seed=2/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_dropout.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_dropout.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline_gamma=.05/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_evidential.yaml -p ./outputs_bin/nuscenes/barrier/cvt_evidential/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_evidential.yaml -p ./outputs_bin/nuscenes/barrier/cvt_evidential_gamma=.05/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_baseline_ol=.0001/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_baseline.yaml -p ./outputs_bin/nuscenes/barrier/cvt_energy_gamma=.05_ol=.0001/19.pt --ep_mode energy -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_evidential.yaml -p ./outputs_bin/nuscenes/barrier/cvt_evidential_ol=.01/19.pt -g 0 1
python eval_barrier.py ./configs/eval_nuscenes_cvt_evidential.yaml -p ./outputs_bin/nuscenes/barrier/cvt_evidential_gamma=.05_ol=.01/19.pt -g 0 1