python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline/19.pt  --ep_mode entropy -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt --ep_mode entropy -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_ensemble.yaml -e ./outputs_bin/carla/vehicle/cvt_baseline/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_seed=1/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_seed=2/19.pt
python eval_ood2.py ./configs/eval_carla_cvt_ensemble.yaml -e ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05_seed=1/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05_seed=2/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_dropout.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_dropout.yaml -p ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_evidential.yaml -p ./outputs_bin/carla/vehicle/cvt_uce_beta=.001/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_evidential.yaml -p ./outputs_bin/carla/vehicle/cvt_ufocal_gamma=.05/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/aug/cvt_energy_ol=.0001/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_baseline.yaml -p ./outputs_bin/carla/aug/cvt_energy_gamma=.05_ol=.0001/19.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_evidential.yaml -p ./outputs_bin/carla/aug/cvt_uce_ol=.01_k=0/15.pt -g 0 1
python eval_ood2.py ./configs/eval_carla_cvt_evidential.yaml -p ./outputs_bin/carla/aug/cvt_ufocal_gamma=.05_ol=.01_k=64/19.pt -g 0 1