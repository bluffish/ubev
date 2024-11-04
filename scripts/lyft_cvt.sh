python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline/19.pt --ep_mode entropy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline_gamma=.05/19.pt --ep_mode entropy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline_gamma=.05/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_ensemble.yaml -e ./outputs/lyft/cvt_baseline/19.pt ./outputs/lyft/cvt_baseline_seed=1/19.pt ./outputs/lyft/cvt_baseline_seed=2/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_ensemble.yaml -e ./outputs/lyft/cvt_baseline_gamma=.05/19.pt ./outputs/lyft/cvt_baseline_gamma=.05_seed=1/19.pt ./outputs/lyft/cvt_baseline_gamma.05_seed=2/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_dropout.yaml -p ./outputs/lyft/cvt_baseline/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_dropout.yaml -p ./outputs/lyft/cvt_baseline_gamma=.05/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_evidential.yaml -p ./outputs/lyft/cvt_evidential/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_evidential.yaml -p ./outputs/lyft/cvt_evidential_gamma=.05/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_baseline.yaml -p ./outputs/lyft/cvt_baseline_gamma=.05_ol=.0001/19.pt --ep_mode energy -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_evidential.yaml -p ./outputs/lyft/cvt_evidential_ol=.01/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_evidential.yaml -p ./outputs/lyft/cvt_evidential_gamma=.05_ol=.01/19.pt -g 4 5
python eval_lyft.py ./configs/eval_lyft_cvt_evidential.yaml -p ./outputs/lyft/cvt_evidential_gamma=.05_ol=.01_k=64/19.pt -g 4 5
