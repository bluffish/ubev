import matplotlib.pyplot as plt
import torch
import numpy as np
from statistics import mean
from tools.metrics import *
from tools.utils import *
from eval import eval


def get_k(aps, k, top=True):
    idxs = list(enumerate(aps))
    s = sorted(idxs, key=lambda x: x[1], reverse=top)
    return s[:k]


def draw(name, set, pred, gt, ep, ood, topk, k=10, top=True):
    fig, axs = plt.subplots(4, k, figsize=(4*k, 20))

    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for i in range(k):
        idx, ap = topk[i]

        axs[0, i].set_title(f"AP={ap:.5f}", fontsize=24)
        axs[0, i].imshow(ep[idx, 0], cmap='inferno')

        axs[1, i].set_title(f"OOD ratio={ood[idx].numpy().mean():.5f}", fontsize=24)
        axs[1, i].imshow(ood[idx], cmap='viridis')

        iou = get_iou(pred[idx].unsqueeze(0), gt[idx].unsqueeze(0), exclude=ood[idx].unsqueeze(0))

        axs[2, i].set_title(f"Avg. IOU={mean(iou):.5f}", fontsize=24)
        axs[2, i].imshow(map_rgb(pred[idx], ego=True) / 255)
        axs[3, i].imshow(map_rgb(gt[idx], ego=True) / 255)

    axs[0, 0].set_ylabel("Epistemic", fontsize=24)
    axs[1, 0].set_ylabel("OOD", fontsize=24)
    axs[2, 0].set_ylabel("Prediction", fontsize=24)
    axs[3, 0].set_ylabel("Label", fontsize=24)

    fig.suptitle(f"{name} on {set}\n Top {k} samples with {'highest' if top else 'lowest'} AP\nID classes are ["
                 f"vehicle, road, lane, background]\nRows are [epistemic, ood, prediction, label]\n", fontsize=24)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    with open('./configs/eval_carla_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['ood'] = True
    config['tsne'] = False
    config['gpus'] = [3, 4, 5, 6]
    config['binary'] = False

    sets = {
        "train": "Train-Pseudo",
        "val": "Val-Pseudo",
        "ood": "OOD-True",
    }

    models = {
        "LSS_UCE_4CLASS": "./outputs/carla/aug/lss_uce_ol=.1_k=0/19.pt",
        # "LSS_UCE_2CLASS": "./outputs_bin/carla/aug/lss_uce_ol=.1_k=0/19.pt",
    }

    # losses = ["UCE", "UFocal"]
    # ols = [".01", ".1", "1"]
    # vacs = ["0", "8", "32", "64"]
    #
    # for loss in losses:
    #     for ol in ols:
    #         for vac in vacs:
    #             models[f"LSS-{loss}-OODReg={ol}-Vac={vac}"] = f"./outputs/grid/{loss.lower()}_ol={ol}_k={vac}/19.pt"

    print(models)

    k = 20

    for lset in sets:
        set_name = sets[lset]
        for name in models:
            path = models[name]

            kset = None
            if lset == 'val' or lset == 'train':
                config['pseudo'] = True
                kset = lset
            if lset == 'ood':
                kset = 'val'
                config['pseudo'] = False

            config['pretrained'] = path

            pred, gt, ood, al, ep, raw = eval(config, kset, "mini", "../data/carla")
            aps = []

            for i in range(ep.shape[0]):
                a = ood[i]
                b = ep[i, 0]

                if a.sum() < 5:
                    continue

                fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(b, a)
                aps.append(ap)

            os.makedirs(f"outputs/ext/{lset}/", exist_ok=True)
            f = draw(name, set_name, pred, gt, ep, ood, get_k(aps, k, top=True), top=True, k=k)
            f.savefig(f"outputs/ext/{lset}/{name}_top.png")
            plt.close(f)
            f = draw(name, set_name, pred, gt, ep, ood, get_k(aps, k, top=False), top=False, k=k)
            f.savefig(f"outputs/ext/{lset}/{name}_bottom.png")
            plt.close(f)
