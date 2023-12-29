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
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    with open('./configs/eval_carla_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['ood'] = True
    config['tsne'] = False
    config['gpus'] = [3, 4, 5, 6]

    sets = {
        "train": "Train-Pseudo",
        "val": "Val-Pseudo",
        "ood": "OOD-True",
    }

    models = {}

    losses = ["UCE", "UFocal"]
    ols = [".01", ".1", "1"]
    vacs = ["0", "8", "32", "64"]

    for loss in losses:
        for ol in ols:
            for vac in vacs:
                models[f"LSS-{loss}-OODReg={ol}-Vac={vac}"] = f"./outputs/grid/{loss.lower()}_ol={ol}_k={vac}/"

    epochs = np.linspace(0, 19, 20)
    tag = ["Vehicle mIOU", "Road mIOU", "Lane mIOU", "Background mIOU", "OOD AUROC", "OOD AP", "Avg. Evidence OOD",
           "Avg. Evidence ID"]

    for name in models:
        path = models[name]

        fig, axs = plt.subplots(2, 4, figsize=(26, 12))
        fig.suptitle(f"Metrics for {name}\nEpochs [0, 19]\nTrain, Val, & Testing set metrics", fontsize=12)

        for i, ax in enumerate(axs.flatten()):
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(tag[i], fontsize=10)
            ax.set_xticks(epochs)
            ax.set_xlim(-.5, 19.5)
            # ax.set_ylim(-.05, 1.05)

            ax.tick_params(axis='both', which='both', bottom=True, left=True, labelsize=8)

        for lset in sets:
            set_name = sets[lset]

            dl = sorted_alphanumeric(os.listdir(path))

            kset = None
            if lset == 'val' or lset == 'train':
                config['pseudo'] = True
                kset = lset
            if lset == 'ood':
                kset = 'val'
                config['pseudo'] = False

            ious = [[], [], [], []]
            aps = []
            aurocs = []
            ep_ood = []
            ep_id = []

            for ch in dl:
                if ch.endswith(".pt"):
                    pre = os.path.join(path, ch)
                    config['pretrained'] = pre

                    pred, gt, ood, al, ep, raw = eval(config, kset, "mini", "../data/carla")

                    iou = get_iou(pred, gt)

                    ep = ep.squeeze(1)
                    ood = ood.bool()

                    _, _, _, _, auroc, ap, _ = roc_pr(ep, ood)

                    for i in range(4):
                        ious[i].append(iou[i])
                    aps.append(ap)
                    aurocs.append(auroc)
                    ep_ood.append(raw.sum(dim=1)[ood.bool()].mean())
                    ep_id.append(raw.sum(dim=1)[~ood.bool()].mean())

            for i in range(4):
                axs[0, i].plot(ious[i], 'o-', label=f"{lset}")

            axs[1, 0].plot(aurocs, 'o-', label=f"{lset}")
            axs[1, 1].plot(aps, 'o-', label=f"{lset}")
            axs[1, 2].plot(ep_ood, 'o-', label=f"{lset}")
            axs[1, 3].plot(ep_id, 'o-', label=f"{lset}")

        for ax in axs.flatten():
            ax.legend()

        fig.savefig(f"outputs/grid_out/{name}_{lset}.png")
        plt.clf()

