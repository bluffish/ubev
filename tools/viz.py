import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from tools.metrics import *


def plot_ece(preds, labels, title=None, exclude=None, n_bins=20):
    conf, acc, ece = expected_calibration_error(preds, labels, exclude=exclude, n_bins=n_bins)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    lower_bin_boundary = bin_boundaries[:-1]
    upper_bin_boundary = bin_boundaries[1:]
    mid_bins = (upper_bin_boundary + lower_bin_boundary) / 2

    ax.bar(mid_bins, acc, alpha=0.7, lw=1, ec='black', fc='#2233aa', width=1. / n_bins,
           label=f'ECE - {ece:.5f}', zorder=0)
    ax.plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2, alpha=.7, color='gray',
            label='Perfectly Calibrated', zorder=1)
    ax.scatter(mid_bins[acc > 0], acc[acc > 0], lw=2, ec='black', fc="#ffffff", zorder=2)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, ece


def plot_roc_pr(y_score, y_true, title=None, exclude=None):
    fpr, tpr, rec, pr, auroc, aupr, no_skill = roc_pr(y_score, y_true, exclude=exclude)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plot_roc(axs[0], fpr, tpr, auroc)
    plot_pr(axs[1], rec, pr, auroc, no_skill)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, auroc, aupr


def plot_roc(ax, fpr, tpr, auroc):
    ax.plot(fpr, tpr, 'b-', label=f'AUROC={auroc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='No Skill - 0.500')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.tick_params(axis='x', which='both', bottom=True)
    ax.tick_params(axis='y', which='both', left=True)
    ax.legend()


def plot_pr(ax, rec, pr, aupr, no_skill):
    ax.step(rec, pr, '-', where='post', label=f'AUPR={aupr:.3f}')
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label=f'No Skill - {no_skill:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.tick_params(axis='x', which='both', bottom=True)
    ax.tick_params(axis='y', which='both', left=True)
    ax.legend()


def add_labels(x, y, ax):
    for i in range(len(x)):
        ax.text(i, y[i], f"{y[i]:.3f}", ha='center')
