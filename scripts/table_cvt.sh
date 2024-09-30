python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --ep_mode entropy --c_pre ./outputs_bin/carla/vehicle/cvt_baseline/19.pt --n_pre ./outputs/nuscenes/cvt_baseline/19.pt
python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --ep_mode entropy --c_pre ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt --n_pre ./outputs/nuscenes/cvt_baseline_gamma=.05/19.pt
python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --ep_mode energy --c_pre ./outputs_bin/carla/vehicle/cvt_baseline/19.pt --n_pre ./outputs/nuscenes/cvt_baseline/19.pt
python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --ep_mode energy --c_pre ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt --n_pre ./outputs/nuscenes/cvt_baseline_gamma=.05/19.pt
python eval_table.py ./configs/eval_carla_cvt_ensemble.yaml --c_ensemble ./outputs_bin/carla/vehicle/cvt_baseline/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_seed=1/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_seed=2/19.pt --n_ensemble ./outputs/nuscenes/cvt_baseline/19.pt ./outputs/nuscenes/cvt_baseline_seed=1/19.pt ./outputs/nuscenes/cvt_baseline_seed=2/19.pt
python eval_table.py ./configs/eval_carla_cvt_ensemble.yaml --c_ensemble ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05_seed=1/19.pt ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05_seed=2/19.pt --n_ensemble ./outputs/nuscenes/cvt_baseline_gamma=.05/19.pt ./outputs/nuscenes/cvt_baseline_gamma=.05_seed=1/19.pt ./outputs/nuscenes/cvt_baseline_gamma=.05_seed=2/19.pt
python eval_table.py ./configs/eval_carla_cvt_dropout.yaml --c_pre ./outputs_bin/carla/vehicle/cvt_baseline/19.pt --n_pre ./outputs/nuscenes/cvt_baseline/19.pt
python eval_table.py ./configs/eval_carla_cvt_dropout.yaml --c_pre ./outputs_bin/carla/vehicle/cvt_baseline_gamma=.05/19.pt --n_pre ./outputs/nuscenes/cvt_baseline_gamma=.05/19.pt
python eval_table.py ./configs/eval_carla_cvt_evidential.yaml --c_pre ./outputs_bin/carla/vehicle/cvt_uce_beta=.001/19.pt --n_pre ./outputs/nuscenes/cvt_evidential/19.pt
python eval_table.py ./configs/eval_carla_cvt_evidential.yaml --c_pre ./outputs_bin/carla/vehicle/cvt_ufocal_gamma=.05/19.pt --n_pre ./outputs/nuscenes/cvt_evidential_gamma=.05/19.pt
python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --c_pre ./outputs_bin/carla/aug/cvt_energy_ol=.0001/19.pt --n_pre ./outputs/nuscenes/cvt_baseline_ol=.0001/19.pt
python eval_table.py ./configs/eval_carla_cvt_baseline.yaml --c_pre ./outputs_bin/carla/aug/cvt_energy_gamma=.05_ol=.0001/19.pt --n_pre ./outputs/nuscenes/cvt_baseline_gamma=.05_ol=.0001/19.pt
python eval_table.py ./configs/eval_carla_cvt_evidential.yaml --c_pre ./outputs_bin/carla/aug/cvt_uce_ol=.01_k=0/15.pt --n_pre ./outputs/nuscenes/cvt_evidential_ol=.01/19.pt
python eval_table.py ./configs/eval_carla_cvt_evidential.yaml --c_pre ./outputs_bin/carla/aug/cvt_ufocal_gamma=.05_ol=.01_k=64/19.pt --n_pre ./outputs/nuscenes/cvt_evidential_gamma=.05_ol=.01_k=64/19.pt

