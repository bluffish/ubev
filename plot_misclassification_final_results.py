import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import torch
import numpy as np
import tqdm
from sklearn.metrics import auc

from eval import eval
from tools.metrics import get_iou, roc_pr, patch_metrics, patch_metrics_q, expected_calibration_error

sns.set_style('white')
sns.set_palette('muted')
sns.set_context(
    "notebook",
    font_scale=1.25,
    rc={"lines.linewidth": 2.5}
)

def plot_misclassification_detection_final_results(
        pt_path, model_name, pos_class,
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
        config['pos_class'] = pos_class

        fig, axs = plt.subplots(9, 3, figsize=(18, 54))
        plt.subplots_adjust(left=0.05, right=0.95)
        for i, lset in enumerate(["test", "val", "train"]):

            preds, labels, oods, aleatoric, epistemic, raws = eval(config, lset, split, dataroot)
            iou = get_iou(preds, labels)

            uncertainty_scores = aleatoric.squeeze(1)
            uncertainty_labels = torch.argmax(labels, dim=1).cpu() != torch.argmax(preds, dim=1).cpu()

            axs[i*3][0].set_ylabel(f'{lset} ROC')
            axs[i*3][1].set_ylabel(f'{lset} PR')
            fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(uncertainty_scores, uncertainty_labels, exclude=oods)
            axs[i*3][0].plot(fpr, tpr, '-', label=f'{model_name}: {auroc:.3f}')
            axs[i*3][1].step(rec, pr, '-', where='post', label=f'{model_name}: {ap:.3f}')
            axs[i*3][0].legend(frameon=True, title="AUROC")
            axs[i*3][1].legend(frameon=True, title="AUPR")

            n_bins = 20
            conf, acc, ece = expected_calibration_error(preds, labels, exclude=None, n_bins=n_bins)
            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            lower_bin_boundary = bin_boundaries[:-1]
            upper_bin_boundary = bin_boundaries[1:]
            mid_bins = (upper_bin_boundary + lower_bin_boundary) / 2
            axs[i*3][2].bar(mid_bins, acc, alpha=0.7, lw=1, ec='black', fc='#2233aa', width=1. / n_bins,
                label=f'ECE - {ece:.5f}', zorder=0)
            axs[i*3][2].plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2, alpha=.7, color='gray',
                label='Perfectly Calibrated', zorder=1)
            axs[i*3][2].scatter(mid_bins[acc > 0], acc[acc > 0], lw=2, ec='black', fc="#ffffff", zorder=2)

            axs[i*3][2].set_xlabel('Confidence')
            axs[i*3][2].set_ylabel('Accuracy')
            axs[i*3][2].set_xlim(0.5, 1.0)
            axs[i*3][2].set_ylim(0.0, 1.0)
            axs[i*3][2].legend()

            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            axs[i*3+1][0].plot(thresholds, agc, '.-', label=f"{model_name}: {au_agc:.3f}")
            axs[i*3+1][1].plot(thresholds, ugi, '.-', label=f"{model_name}: {au_ugi:.3f}")
            axs[i*3+1][2].plot(thresholds, pavpu, '.-', label=f"{model_name}: {au_pavpu:.3f}")
            axs[i*3+1][0].legend(frameon=True, title="AU-p(accurate|certain)")
            axs[i*3+1][0].set_xlabel("threshold")
            axs[i*3+1][0].set_ylabel("p(accurate|certain)")
            axs[i*3+1][1].legend(frameon=True, title="AU-p(uncertain|inaccurate)")
            axs[i*3+1][0].set_xlabel("threshold")
            axs[i*3+1][0].set_ylabel("p(uncertain|inaccurate)")
            axs[i*3+1][2].legend(frameon=True, title="AU-PAvPU")
            axs[i*3+1][0].set_xlabel("threshold")
            axs[i*3+1][0].set_ylabel("PAvPU")

            pavpu, agc, ugi, thresholds, percs, au_pavpu, au_agc, au_ugi = patch_metrics_q(uncertainty_scores,
                                                                                  uncertainty_labels)
            axs[i*3+2][0].plot(percs, agc, '.-', label=f"{model_name}: {au_agc:.3f}")
            axs[i*3+2][1].plot(percs, ugi, '.-', label=f"{model_name}: {au_ugi:.3f}")
            axs[i*3+2][2].plot(percs, pavpu, '.-', label=f"{model_name}: {au_pavpu:.3f}")
            axs[i*3+2][0].legend(frameon=True, title="AU-p(accurate|certain)")
            axs[i*3+2][0].set_xlabel("percentile")
            axs[i*3+2][0].set_ylabel("p(accurate|certain)")
            axs[i*3+2][1].legend(frameon=True, title="AU-p(uncertain|inaccurate)")
            axs[i*3+2][0].set_xlabel("percentile")
            axs[i*3+2][0].set_ylabel("p(uncertain|inaccurate)")
            axs[i*3+2][2].legend(frameon=True, title="AU-PAvPU")
            axs[i*3+2][0].set_xlabel("percentile")
            axs[i*3+2][0].set_ylabel("PAvPU")

        for i in range(len(axs)):
            for j in range(len(axs[i])):
                axs[i][j].set_xlim([-0.05, 1.05])
                axs[i][j].set_ylim([-0.05, 1.05])
    
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{model_name}-ValidationTestTrain-mis-Metrics.png"), format='png') 
    fig.savefig(os.path.join(save_path, f"{model_name}-ValidationTestTrain-mis-Metrics.svg"), format='svg')

if __name__ == "__main__":
    for pos_class in ["vehicle", "lane", "road"]:
        models_folder = f"outputs_bin/carla/{pos_class}"
        for model_folder_name in tqdm.tqdm(os.listdir(models_folder)):
            pt_path = os.path.join(models_folder, model_folder_name, "19.pt")
            if not os.path.exists(pt_path):
                continue
            if model_folder_name.startswith("lss"):
                config_path="./configs/eval_carla_lss_evidential.yaml"
            elif model_folder_name.startswith("cvt"):
                config_path="./configs/eval_carla_cvt_evidential.yaml"
            model_name = pos_class+"_"+model_folder_name
            plot_misclassification_detection_final_results(pt_path=pt_path, model_name=model_name, config_path=config_path, pos_class=pos_class, save_path=f"plots/carla_{pos_class}")
