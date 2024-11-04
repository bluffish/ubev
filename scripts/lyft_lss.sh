python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline/19.pt --ep_mode entropy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline_gamma=1/19.pt --ep_mode entropy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline_gamma=1/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_ensemble.yaml -e ./outputs/lyft/lss_baseline/19.pt ./outputs/lyft/lss_baseline_seed=1/19.pt ./outputs/lyft/lss_baseline_seed=2/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_ensemble.yaml -e ./outputs/lyft/lss_baseline_gamma=1/19.pt ./outputs/lyft/lss_baseline_gamma=1_seed=1/19.pt ./outputs/lyft/lss_baseline_gamma=1_seed=2/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_dropout.yaml -p ./outputs/lyft/lss_baseline/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_dropout.yaml -p ./outputs/lyft/lss_baseline_gamma=1/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_evidential.yaml -p ./outputs/lyft/lss_evidential/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_evidential.yaml -p ./outputs/lyft/lss_evidential_gamma=.5/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_baseline.yaml -p ./outputs/lyft/lss_baseline_gamma=1_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_evidential.yaml -p ./outputs/lyft/lss_evidential_ol=.01/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_evidential.yaml -p ./outputs/lyft/lss_evidential_gamma=.5_ol=.01/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_lss_evidential.yaml -p ./outputs/lyft/lss_evidential_gamma=.5_ol=.01_k=128/19.pt -g 4 5