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
    parser.add_argument("metric")
    parser.add_argument("set")

    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-l', '--logdir', default="./outputs", required=False)
    parser.add_argument('-g', '--gpus', nargs='+', required=False)
    parser.add_argument('-s', '--lset', default='val', required=False)
    parser.add_argument('-t', '--title', required=False)
    parser.add_argument('--pseudo', default=False, action='store_true')

    args = parser.parse_args()

    with open(args.set, 'r') as file:
        set = yaml.safe_load(file)

    set_name = args.set.split('.')[-2].split('/')[-1]
    names = list(set.keys())
    logdir = args.logdir

    os.makedirs(logdir, exist_ok=True)

    metric = args.metric
    is_ood = args.ood
    lset = args.lset
    title = args.title

    scale = 1.5

    if metric == 'patch':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95)
        ax1.set_xlabel('Uncertainty Percentiles')
        ax1.set_ylabel('p(accurate|certain)')

        ax2.set_xlabel('Uncertainty Percentiles')
        ax2.set_ylabel('p(uncertain|inaccurate)')

        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
    elif metric == 'prob':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*scale, 6*scale))
        plt.subplots_adjust(left=0.05, right=0.95)

        ax1.set_xlabel('Uncertainty Threshold')
        ax1.set_ylabel('p(accurate|certain)')

        ax2.set_xlabel('Uncertainty Threshold')
        ax2.set_ylabel('p(accurate|certain)')

    elif metric == 'rocpr':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*scale, 6*scale))

        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
    elif metric == 'all':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(left=0.05, right=0.95)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')

        ax3.set_xlabel('Uncertainty Percentiles')
        ax3.set_ylabel('PAVPU')
    else:
        raise ValueError("Please pick a valid metric.")

    iou_path = f"{logdir}/iou_{'o' if is_ood else 'm'}_{set_name}.txt"

    no_skill_total = 0

    for name in names:
        torch.manual_seed(0)
        np.random.seed(0)

        with open(set[name]['config'], 'r') as file:
            config = yaml.safe_load(file)
            config['pretrained'] = set[name]['path']
            config['logdir'] = f"outputs/graph/{name}"
            config['three'] = config['five'] = config['tsne'] = False

            print(config['pretrained'])
            if args.gpus is not None:
                config['gpus'] = [int(i) for i in args.gpus]

        split = "mini"
        dataroot = f"../data/{config['dataset']}"
        config['ood'] = True
        config['pseudo'] = args.pseudo
        config['binary'] = True

        predictions, ground_truth, oods, aleatoric, epistemic, raws = eval(config, lset, split, dataroot)

        label = set[name]['label'] if 'label' in set[name] else name

        iou = get_iou(predictions, ground_truth)
        print(f"mIOU: {iou}")

        with open(iou_path, 'a') as file:
            file.write(f"{label}: {[round(i, 3) for i in iou]}\n")

        if is_ood:
            uncertainty_scores = epistemic.squeeze(1)
            uncertainty_labels = oods
        else:
            uncertainty_scores = aleatoric.squeeze(1)
            uncertainty_labels = torch.argmax(ground_truth, dim=1).cpu() != torch.argmax(predictions, dim=1).cpu()

        if metric == 'patch':
            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            ax1.plot(thresholds, agc, '.-', label=f"{label}: {au_agc:.3f}")
            ax2.plot(thresholds, ugi, '.-', label=f"{label}: {au_ugi:.3f}")
            ax3.plot(thresholds, pavpu, '.-', label=f"{label}: {au_pavpu:.3f}")

            print(f"AU-PAvPU - {au_pavpu:.3f}, AU-p(accurate|certain) - {au_agc:.3f}, AU-P(uncertain|inaccurate) - {au_ugi:.3f}")
        elif metric == "prob":
            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            ax1.plot(thresholds, agc, '.-', label=f"{label}: {au_agc:.3f}")
            ax2.plot(thresholds, ugi, '.-', label=f"{label}: {au_ugi:.3f}")

            print(f"AU-p(accurate|certain) - {au_agc:.3f}, AU-P(uncertain|inaccurate) - {au_ugi:.3f}")
        elif metric == "rocpr":
            fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(uncertainty_scores, uncertainty_labels, exclude=oods)

            ax1.plot(fpr, tpr, '-', label=f'{label}: {auroc:.3f}')
            ax2.step(rec, pr, '-', where='post', label=f'{label}: {ap:.3f}')

            no_skill_total += no_skill

            print(f"AUROC: {auroc:.3f}, AP: {ap:.3f}")
        elif metric == "all":
            fpr, tpr, rec, pr, auroc, ap, no_skill = roc_pr(uncertainty_scores, uncertainty_labels)

            pavpu, agc, ugi, thresholds, au_pavpu, au_agc, au_ugi = patch_metrics(uncertainty_scores,
                                                                                  uncertainty_labels)

            ax1.plot(fpr, tpr, '-', label=f'{label}: {auroc:.3f}')
            ax2.step(rec, pr, '-', where='post', label=f'AP - {ap:.3f}')
            ax3.plot(thresholds, pavpu, '.-', label=f"{label}: {au_pavpu:.3f}")

            print(f"AUROC: {auroc:.3f}, AP: {ap:.3f}, AU-PAvPU - {au_pavpu:.3f}")

    if metric == 'all':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax3.set_xlim([-0.05, 1.05])
        ax3.set_ylim([-0.05, 1.05])

        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax2.plot([0, 1], [no_skill_total / len(names), no_skill_total / len(names)], linestyle='--', color='gray', label=f'No Skill: {no_skill:.3f}')

        ax1.legend(frameon=True)
        ax2.legend(frameon=True)
        ax3.legend(frameon=True)
    elif metric == 'patch':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax3.set_xlim([-0.05, 1.05])
        ax3.set_ylim([-0.05, 1.05])
        ax1.legend(frameon=True)
        ax2.legend(frameon=True)
        ax3.legend(frameon=True)
    elif metric == 'prob':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])

        ax1.legend(frameon=True)
        ax2.legend(frameon=True)
    elif metric == 'rocpr':
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax2.set_xlim([-0.05, 1.05])
        ax2.set_ylim([-0.05, 1.05])
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
        ax2.plot([0, 1], [no_skill_total / len(names), no_skill_total / len(names)], linestyle='--', color='gray', label=f'No Skill: {no_skill:.3f}')

        ax1.legend(frameon=True, title="AUROC")
        ax2.legend(frameon=True, title='AUPR')

    if title is None:
        fig.suptitle(f"{'OOD' if is_ood else 'Misclassification'}")
    else:
        fig.suptitle(title)

    save_path = f"{logdir}/{metric}_{'o' if is_ood else 'm'}_{set_name}"
    fig.savefig(save_path)
    fig.savefig(f"{save_path}.pdf", format='pdf')
