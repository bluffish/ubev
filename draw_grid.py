import matplotlib.pyplot as plt
import torch
import numpy as np
from statistics import mean
from tools.metrics import *
from tools.utils import *
from eval import eval
import re

from tools.viz import plot_roc_pr


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('-?([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":
    m = Evidential([0], backbone="lss")

    with open('./configs/eval_carla_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['ood'] = True
    config['gpus'] = [5, 6]
    config['pos_class'] = 'vehicle'

    tags = ["Positive Class mIOU", "Background mIOU", "OOD AUPR", "OOD AUROC", "mIOU AUPR Avg.",
            "Total Loss", "OOD Reg.", "UCE/UFocal Loss"]

    sets = {
        "val_aug_stable": "Validation set w/ Pseudo OOD",
        "ood": "Test set w/ True OOD",
        "train_aug_stable": "Training set w/ Pseudo OOD",
    }

    epochs = np.linspace(0, 19, 20)

    models = {
        # "LSS_UCE_OOD-Reg=.01_Vac=0": "./outputs_bin/carla/grid_aug/uce_ol=.01_k=0",
        # "LSS_UCE_OOD-Reg=.01_Vac=16": "./outputs_bin/carla/grid_aug/uce_ol=.01_k=16",
        # "LSS_UCE_OOD-Reg=.01_Vac=32": "./outputs_bin/carla/grid_aug/uce_ol=.01_k=32",
        # "LSS_UCE_OOD-Reg=.01_Vac=64": "./outputs_bin/carla/grid_aug/uce_ol=.01_k=64",
        # "LSS_UCE_OOD-Reg=.1_Vac=0": "./outputs_bin/carla/grid_aug/uce_ol=.1_k=0",
        # "LSS_UCE_OOD-Reg=.1_Vac=16": "./outputs_bin/carla/grid_aug/uce_ol=.1_k=16",
        # "LSS_UCE_OOD-Reg=.1_Vac=32": "./outputs_bin/carla/grid_aug/uce_ol=.1_k=32",
        # "LSS_UCE_OOD-Reg=.1_Vac=64": "./outputs_bin/carla/grid_aug/uce_ol=.1_k=64",
        # "LSS_UCE_OOD-Reg=1_Vac=0": "./outputs_bin/carla/grid_aug/uce_ol=1_k=0",
        # "LSS_UCE_OOD-Reg=1_Vac=16": "./outputs_bin/carla/grid_aug/uce_ol=1_k=16",
        # "LSS_UCE_OOD-Reg=1_Vac=32": "./outputs_bin/carla/grid_aug/uce_ol=1_k=32",
        # "LSS_UCE_OOD-Reg=1_Vac=64": "./outputs_bin/carla/grid_aug/uce_ol=1_k=64",
        # "LSS_UFocal_OOD-Reg=.01_Vac=64": "./outputs_bin/carla/grid_aug/ufocal_ol=.01_k=64",
        # "CVT_UFocal_gamma=.05_OOD-Reg=.01_Vac=64": "./outputs_bin/carla/grid_aug/cvt_ufocal_gamma=.05_ol=.01_k=64",
        # "CVT_UCE_OOD-Reg=.01_Vac=64": "./outputs_bin/carla/grid_aug/cvt_uce_ol=.01_k=64",
        "StableDiffAug": "./outputs_bin/carla_stablediff_test"
    }

    fig, axs = plt.subplots(6, 4, figsize=(30, 36))

    for si, s in enumerate(sets):
        axs[si * 2, 2].set_title(sets[s], fontsize=16)

        for i in range(len(tags)):
            ax = axs[si * 2 + i // 4, i % 4]
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(tags[i], fontsize=12)
            ax.set_xticks(epochs)
            ax.set_xlim(-.5, 19.5)

            ax.tick_params(axis='both', which='both', bottom=True, left=True, labelsize=10)

    fig.suptitle("Comparison of All Models", fontsize=24)

    for name in models:
        torch.manual_seed(0)
        np.random.seed(0)

        fig2, axs2 = plt.subplots(3, 2, figsize=(3 * 6, 3 * 6))
        fig2.suptitle(name)
        config['pretrained'] = os.path.join(models[name], "19.pt")

        for i, set in enumerate(sets):
            preds, labels, oods, aleatoric, epistemic, raw = eval(config, set, "mini", "../data/carla")

            plot_roc_pr(epistemic, oods, axs=(axs2[i, 0], axs2[i, 1]))
            axs2[i, 0].set_title(f"{sets[set]} ROC Curve")
            axs2[i, 1].set_title(f"{sets[set]} PR Curve")

        fig2.tight_layout()
        fig2.savefig(os.path.join('outputs_bin', f"{name}_ood_metrics.png"))
        fig2.savefig(os.path.join('outputs_bin', f"{name}_ood_metrics.pdf"), format='pdf')

        plt.close(fig2)

        checkpoints = sorted_alphanumeric(os.listdir(models[name]))

        for si, s in enumerate(sets):
            set_name = s

            data = [[] for _ in range(len(tags))]

            for checkpoint in checkpoints:
                if checkpoint.endswith(".pt"):
                    config['pretrained'] = os.path.join(models[name], checkpoint)

                    pred, gt, ood, al, ep, raw = eval(config, set_name, "mini", "../data/carla")

                    ious = get_iou(pred, gt)
                    ep = ep.squeeze(1)
                    ood = ood.bool()
                    _, _, _, _, auroc, aupr, _ = roc_pr(ep, ood)
                    total, ood_reg = m.loss_ood(raw, gt, ood.long())
                    uce = total - ood_reg

                    data[2].append(aupr)
                    data[3].append(auroc)
                    data[4].append((ious[0] + aupr) / 2.)
                    data[5].append(total)
                    data[6].append(ood_reg)
                    data[7].append(uce)

                    for i in range(len(ious)):
                        data[i].append(ious[i])

            for i in range(len(data)):
                ax = axs[si * 2 + i // 4, i % 4]
                ax.plot(data[i], 'o-', label=name)

    for ax in axs.flatten():
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig.savefig(f"outputs_bin/All-TrainValidationTest-ood-Curves.png")
    fig.savefig(f"outputs_bin/All-TrainValidationTest-ood-Curves.pdf", format="pdf")

    plt.clf()

