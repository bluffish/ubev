python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline/19.pt --ep_mode entropy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline_gamma=.05/19.pt --ep_mode entropy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline/19.pt --ep_mode energy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline_gamma=.05/19.pt --ep_mode energy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_ensemble.yaml -e ./outputs/nuscenes/simplebev_baseline/19.pt ./outputs/nuscenes/simplebev_baseline_seed=1/19.pt ./outputs/nuscenes/simplebev_baseline_seed=2/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_ensemble.yaml -e ./outputs/nuscenes/simplebev_baseline_gamma=.05/19.pt ./outputs/nuscenes/simplebev_baseline_gamma=.05_seed=1/19.pt ./outputs/nuscenes/simplebev_baseline_gamma=.05_seed=2/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_dropout.yaml -p ./outputs/nuscenes/simplebev_baseline/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_dropout.yaml -p ./outputs/nuscenes/simplebev_baseline_gamma=.05/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_evidential.yaml -p ./outputs/nuscenes/simplebev_evidential/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_evidential.yaml -p ./outputs/nuscenes/simplebev_evidential_gamma=.05/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_baseline.yaml -p ./outputs/nuscenes/simplebev_baseline_gamma=.05_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_evidential.yaml -p ./outputs/nuscenes/simplebev_evidential_ol=.01/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_evidential.yaml -p ./outputs/nuscenes/simplebev_evidential_gamma=.05_ol=.01/19.pt -g 4 5
python eval_nusc.py ./configs/eval_nuscenes_simplebev_evidential.yaml -p ./outputs/nuscenes/simplebev_evidential_gamma=.05_ol=.01_k=64/19.pt -g 4 5
