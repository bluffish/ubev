import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tools.metrics import *

from eval import eval
from tools.utils import *
from tensorboardX import SummaryWriter

import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


if __name__ == "__main__":
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
                models[f"LSS-{loss}-OODReg={ol}-Vac={vac}"] = f"./outputs/grid/{loss.lower()}_ol={ol}_k={vac}/19.pt"

    with open('./configs/eval_carla_fiery_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    split = "mini"
    dataroot = f"../data/carla"
    path = "outputs/aug"

    for s in sets:
        os.makedirs(f"./{path}/hists_ood/{s}")
        writer = SummaryWriter(logdir=f"./{path}/hists_ood/{s}")

        dl = sorted_alphanumeric(os.listdir(f"./{path}/{s}"))
        for ch in dl:
            if ch.endswith(".pt"):
                pre = os.path.join(f"./{path}/{s}", ch)
                config['pretrained'] = pre
                config['gpus'] = [4, 5, 6, 7]
                config['five'] = False
                config['three'] = False
                config['tsne'] = False
                config['ood'] = True

                torch.manual_seed(0)
                np.random.seed(0)

                predictions, ground_truth, oods, aleatoric, epistemic, raw = eval(config, True, 'val', split, dataroot)
                uncertainty_scores = epistemic.squeeze(1)
                uncertainty_labels = oods.bool()

                unc_iou = get_iou(torch.cat((uncertainty_scores[:, None], 1 - uncertainty_scores[:, None]), dim=1),
                        torch.cat((uncertainty_labels[:, None].long(), (~uncertainty_labels[:, None]).long()), dim=1))
                iou = get_iou(predictions, ground_truth)

                fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)
                e = ece(predictions, ground_truth)

                writer.add_scalar("hist/auroc", auroc, int(ch.split(".")[0]))
                writer.add_scalar("hist/aupr", aupr, int(ch.split(".")[0]))
                writer.add_scalar("hist/ece", e, int(ch.split(".")[0]))
                writer.add_scalar("hist/unc_iou", unc_iou[0], int(ch.split(".")[0]))

                writer.add_scalar("hist/vehicle_iou", iou[0], int(ch.split(".")[0]))
                writer.add_scalar("hist/road_iou", iou[1], int(ch.split(".")[0]))
                writer.add_scalar("hist/lane_iou", iou[2], int(ch.split(".")[0]))
                writer.add_scalar("hist/avt", ((iou[0]+iou[1]+iou[2]+iou[3])/4 + aupr)/2, int(ch.split(".")[0]))

