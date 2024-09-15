python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/vehicle/lss_baseline/19.pt --ep_mode entropy -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5/19.pt --ep_mode entropy -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/vehicle/lss_baseline/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_ensemble.yaml -e ./outputs_bin/carla/vehicle/lss_baseline/19.pt ./outputs_bin/carla/vehicle/lss_baseline_seed=1/19.pt ./outputs_bin/carla/vehicle/lss_baseline_seed=2/19.pt  -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_ensemble.yaml -e ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5/19.pt ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5_seed=1/19.pt ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5_seed=2/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_dropout.yaml -p ./outputs_bin/carla/vehicle/lss_baseline/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_dropout.yaml -p ./outputs_bin/carla/vehicle/lss_baseline_gamma=.5/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_evidential.yaml -p ./outputs_bin/carla/vehicle/lss_uce_beta=.001/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_evidential.yaml -p ./outputs_bin/carla/vehicle/lss_ufocal_gamma=.5/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/aug/lss_energy_ol=.0001/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_baseline.yaml -p ./outputs_bin/carla/aug/lss_energy_gamma=1_ol=.0001/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_evidential.yaml -p ./outputs_bin/carla/aug/lss_uce_ol=.01_k=0/19.pt -g 2 3
python eval_ood2.py ./configs/eval_carla_lss_evidential.yaml -p ./outputs_bin/carla/aug/lss_ufocal_gamma=.1_ol=.01_k=64/19.pt -g 2 3