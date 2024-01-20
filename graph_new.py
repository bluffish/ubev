import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from eval import eval
from tools.utils import *
from tools.metrics import *

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group")

    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-l', '--logdir', default="./outputs", required=False)
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-s', '--set', default='ood', required=False)
    parser.add_argument('--split', default='mini', required=False)
    parser.add_argument('-t', '--title', required=False)
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)

    args = parser.parse_args()

    with open(args.group, 'r') as file:
        group = yaml.safe_load(file)

    set_name = args.group.split('.')[-2].split('/')[-1]
    names = list(group.keys())
    logdir = args.logdir

    os.makedirs(logdir, exist_ok=True)

    is_ood = args.ood
    set = args.set
    title = args.title

    scale = 1.5
    n_bins = 50

    fig, axs = plt.subplots(1, 3, figsize=(18*scale, 6*scale))

    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[2].set_xlabel('Confidence')
    axs[2].set_ylabel('Accuracy')

    no_skill_total = 0

    for name in names:
        torch.manual_seed(0)
        np.random.seed(0)

        with open(group[name]['config'], 'r') as file:
            config = yaml.safe_load(file)
            config['pretrained'] = group[name]['path']
            config['logdir'] = f"{logdir}/graph/{name}"
            config['pos_class'] = args.pos_class

            print(config['pretrained'])
            if args.gpus is not None:
                config['gpus'] = [int(i) for i in args.gpus]

        split = args.split
        dataroot = f"../data/{config['dataset']}"

        preds, labels, oods, aleatoric, epistemic, raw = eval(config, set, split, dataroot)
        label = group[name]['label'] if 'label' in group[name] else name

        fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(aleatoric, get_mis(preds, labels), exclude=oods)
        conf, acc, ece = expected_calibration_error(preds, labels, exclude=oods, n_bins=n_bins)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        lower_bin_boundary = bin_boundaries[:-1]
        upper_bin_boundary = bin_boundaries[1:]
        mid_bins = (upper_bin_boundary + lower_bin_boundary) / 2

        axs[0].plot(fpr, tpr, '-', label=f'{label}: {auroc:.5f}')
        axs[1].step(rec, pr, '-', where='post', label=f'{label}: {aupr:.5f}')
        axs[2].plot(mid_bins[acc > 0], acc[acc > 0], marker='o', label=f'{label}: {ece:.5f}')

        no_skill_total += no_skill

        print(f"AUROC: {auroc:.3f}, AUPR: {aupr:.3f}, ECE: {ece:.5f}")

    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    axs[2].set_xlim(0.5, 1.0)

    axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
    axs[1].plot([0, 1], [no_skill_total / len(names), no_skill_total / len(names)], linestyle='--', color='gray', label=f'No Skill: {no_skill:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label="Perfectly Calibrated")

    axs[0].legend(frameon=True, title="AUROC")
    axs[1].legend(frameon=True, title='AUPR')
    axs[2].legend(frameon=True, title='ECE')

    if title is None:
        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'}")
    else:
        fig.suptitle(title)

    save_path = f"{logdir}/{'o' if is_ood else 'm'}_{set_name}"
    fig.savefig(save_path)
    fig.savefig(f"{save_path}.pdf", format='pdf')
