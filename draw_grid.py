import matplotlib.pyplot as plt
import torch
import numpy as np
from statistics import mean
from tools.metrics import *
from tools.utils import *
from eval import eval
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('-?([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    m = Evidential([0], backbone="lss")

    with open('./configs/eval_carla_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['ood'] = True
    config['gpus'] = [5, 6]
    config['binary'] = True

    if config['binary']:
        tags = ["Vehicle mIOU", "Road mIOU", "Lane mIOU", "Background mIOU", "OOD mIOU",
                "Total Loss", "OOD Reg.", "UCE/UFocal Loss", "OOD AUPR", "OOD AUROC"]
    else:
        tags = ["Vehicle mIOU", "Background mIOU", "Lane mIOU", "Road mIOU", "OOD mIOU",
                "Total Loss", "OOD Reg.", "UCE/UFocal Loss", "OOD AUPR", "OOD AUROC"]

    sets = {
        "val": "Validation set w/ Pseudo OOD",
        "ood": "Test set w/ True OOD",
        "train": "Training set w/ Pseudo OOD",
    }

    epochs = np.linspace(0, 19, 20)

    models = {
        # "LSS_UCE_Bin_OODReg=.1": "./outputs_bin/carla/aug/lss_uce_ol=.1_k=0",
        # "LSS_UCE_Four_OODReg=.1": "./outputs/carla/aug/lss_uce_ol=.1_k=0",
        "LSS_UFocal_Bin_OODReg=1_Vac=32": "./outputs_bin/carla/aug/lss_ufocal_nopretrain_ol=1_k=32"
    }

    for name in models:
        fig, axs = plt.subplots(6, 5, figsize=(30, 36))

        for si, s in enumerate(sets):
            axs[si * 2, 2].set_title(sets[s], fontsize=16)

            for i in range(len(tags)):
                ax = axs[si * 2 + i // 5, i % 5]
                ax.set_xlabel("Epoch", fontsize=12)
                ax.set_ylabel(tags[i], fontsize=12)
                ax.set_xticks(epochs)
                ax.set_xlim(-.5, 19.5)

                ax.tick_params(axis='both', which='both', bottom=True, left=True, labelsize=10)

        fig.suptitle(name, fontsize=24)

        checkpoints = sorted_alphanumeric(os.listdir(models[name]))

        for si, s in enumerate(sets):
            set_name = s

            if s == 'val' or s == 'train':
                config['pseudo'] = True
            if s == 'ood':
                config['pseudo'] = False
                set_name = "val"

            data = [[] for _ in range(len(tags))]

            for checkpoint in checkpoints:
                if checkpoint.endswith(".pt"):
                    config['pretrained'] = os.path.join(models[name], checkpoint)

                    pred, gt, ood, al, ep, raw = eval(config, set_name, "mini", "../data/carla")

                    ious = get_iou(pred, gt)
                    ep = ep.squeeze(1)
                    ood = ood.bool()
                    _, _, _, _, auroc, ap, _ = roc_pr(ep, ood)
                    total, ood_reg = m.loss_ood(raw, gt, ood.long())
                    uce = total - ood_reg

                    data[5].append(total)
                    data[6].append(ood_reg)
                    data[7].append(uce)
                    data[8].append(ap)
                    data[9].append(auroc)

                    data[4].append(unc_iou(ep, ood, thresh=.5))

                    for i in range(len(ious)):
                        data[i].append(ious[i])

            for i in range(len(data)):
                ax = axs[si * 2 + i // 5, i % 5]
                ax.plot(data[i], 'o-', label=name)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        fig.savefig(f"outputs/{name}-TrainValidationTest-Curves.png")
        fig.savefig(f"outputs/{name}-TrainValidationTest-Curves.pdf", format="pdf")

        plt.clf()

