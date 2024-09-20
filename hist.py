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
        # "train": "Train-Pseudo",
        # "val": "Val-Pseudo",
        "ood": "OOD-True",
    }

    models = {"LSS_UFCE_VS_ER": "./outputs_bin/lyft/ufce_vs_er_long"}

    # losses = ["UCE", "UFocal"]
    # ols = [".01", ".1", "1"]
    # vacs = ["0", "8", "32", "64"]
    #
    # for loss in losses:
    #     for ol in ols:
    #         for vac in vacs:
    #             models[f"LSS-{loss}-OODReg={ol}-Vac={vac}"] = f"./outputs/grid/{loss.lower()}_ol={ol}_k={vac}/19.pt"

    with open('./configs/eval_lyft_lss_evidential.yaml', 'r') as file:
        config = yaml.safe_load(file)

    split = "mini"
    dataroot = f"../data/lyft"
    model = "uce_ent_vs_er"
    path = f"./outputs_bin/lyft/{model}"

    for s in sets:
        os.makedirs(f"./outputs/hists_ood_long/{model}", exist_ok=True)
        writer = SummaryWriter(logdir=f"./outputs/hists_ood_long/{model}")

        dl = sorted_alphanumeric(os.listdir(f"./{path}"))
        for ch in dl:
            if ch.endswith(".pt"):
                pre = os.path.join(f"./{path}/", ch)
                config['pretrained'] = pre
                config['gpus'] = [4, 5]
                config['ood'] = True
                config['pos_class'] = "vehicle"

                torch.manual_seed(0)
                np.random.seed(0)

                predictions, ground_truth, oods, aleatoric, epistemic, raw = eval(config, 'ood', split, dataroot)
                uncertainty_scores = epistemic.squeeze(1)
                uncertainty_labels = oods.bool()

                fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)
                writer.add_scalar("hist/auroc", auroc, int(ch.split(".")[0]))
                writer.add_scalar("hist/aupr", aupr, int(ch.split(".")[0]))

