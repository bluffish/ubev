import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import torch
import numpy as np
import tqdm
import json

from eval import eval
from tools.metrics import get_iou, roc_pr

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)


def replace_last_section(path, new_section):
    head, tail = os.path.split(path)
    new_path = os.path.join(head, new_section)

    return new_path

def plot_ood_detection_final_results(
        pt_path, model_name, pos_class_name,
        config_path="./configs/eval_carla_lss_evidential.yaml",
        pseudo_oods=False,
        gpus=[4,5],
        save_path="plots/",
    ):
    torch.manual_seed(0)
    np.random.seed(0)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        config['pretrained'] = pt_path
        config['logdir'] = f"plots/ood/{model_name}"
        config['three'] = config['five'] = config['tsne'] = False

        if gpus is not None:
            config['gpus'] = [int(i) for i in gpus]
        
        split = "mini"
        dataroot = f"../data/{config['dataset']}"
        config['ood'] = True
        config['pseudo'] = pseudo_oods
        config['binary'] = True
        config['pos_class'] = "vehicle"

    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    plt.subplots_adjust(left=0.05, right=0.95)

    ious = {}

    for i, lset in enumerate(["ood", "val_aug", "train_aug"]):
        predictions, ground_truth, oods, aleatoric, epistemic, raws = eval(config, lset, split, dataroot)

        iou = get_iou(predictions, ground_truth, exclude=oods)
        ious[lset] = iou

        # OOD
        uncertainty_scores = epistemic.squeeze(1)
        uncertainty_labels = oods

        fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

        axs[i][0].plot(fpr, tpr, '-', label=f'{model_name}: {auroc:.3f}')
        axs[i][1].step(rec, pr, '-', where='post', label=f'{model_name}: {ap:.3f}')

        axs[i][0].set_xlim([-0.05, 1.05])
        axs[i][0].set_ylim([-0.05, 1.05])
        axs[i][0].set_title(f"{lset} AUROC")
        axs[i][0].legend(frameon=True)
        axs[i][1].set_xlim([-0.05, 1.05])
        axs[i][1].set_ylim([-0.05, 1.05])
        axs[i][1].set_title(f"{lset} AUPR")
        axs[i][1].legend(frameon=True)

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{model_name}-ood-Metrics.png"), format='png')
    fig.savefig(os.path.join(save_path, f"{model_name}-ood-Metrics.svg"), format='svg')
    with open(os.path.join(save_path, f"{model_name}-ious.json"), 'w') as f:
        json.dump(ious, f)

if __name__ == "__main__":
    models_folder = "outputs_bin/carla/grid_aug"
    for model_folder_name in tqdm.tqdm(os.listdir(models_folder)):
        if model_folder_name.startswith("cvt"):
            continue
        pt_path = os.path.join(models_folder, model_folder_name, "19.pt")
        model_name = "LSS_CARLA_grid_aug_"+model_folder_name
        plot_ood_detection_final_results(pt_path=pt_path, model_name=model_name, pos_class_name="vehicle", save_path="plots/LSS_CARLA_grid_aug")
