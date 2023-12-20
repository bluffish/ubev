import cv2
import numpy as np
import torch
from time import time

from sklearn.manifold import TSNE
from sklearn.metrics import *
from sklearn.calibration import *
import torchmetrics
import matplotlib.pyplot as plt


def get_iou(preds, labels):
    classes = preds.shape[1]
    iou = [0] * classes

    pmax = preds.argmax(dim=1)
    lmax = labels.argmax(dim=1)

    with torch.no_grad():
        for i in range(classes):
            p = (pmax == i).bool()
            l = (lmax == i).bool()

            intersect = (p & l).sum().float().item()
            union = (p | l).sum().float().item()
            iou[i] = intersect / union if union > 0 else 0

    return iou


def patch_metrics(uncertainty_scores, uncertainty_labels):
    thresholds = np.linspace(0, 1, 11)

    pavpus = []
    agcs = []
    ugis = []

    for thresh in thresholds:
        pavpu, agc, ugi = calculate_pavpu(uncertainty_scores, uncertainty_labels, uncertainty_threshold=thresh)

        pavpus.append(pavpu)
        agcs.append(agc)
        ugis.append(ugi)

    return pavpus, agcs, ugis, thresholds, auc(thresholds, pavpus), auc(thresholds, agcs), auc(thresholds, ugis)


def unc_iou(uncertainty_scores, uncertainty_labels):
    thresholds = np.linspace(0, 1, 11)
    ious = []

    for thresh in thresholds:
        with torch.no_grad():
            label = uncertainty_labels.bool()
            pred = (uncertainty_scores > thresh).bool()
            union = (label | pred)
            ious.append((label & pred) / (union if union > 0 else 1))

    return ious, thresholds


def calculate_pavpu(uncertainty_scores, uncertainty_labels, accuracy_threshold=0.5, uncertainty_threshold=0.2,
                    window_size=1):
    if window_size == 1:
        accurate = ~uncertainty_labels.long()
        uncertain = uncertainty_scores >= uncertainty_threshold

        au = torch.sum(accurate & uncertain)
        ac = torch.sum(accurate & ~uncertain)
        iu = torch.sum(~accurate & uncertain)
        ic = torch.sum(~accurate & ~uncertain)
    else:
        ac, ic, au, iu = 0., 0., 0., 0.

        anchor = (0, 0)
        last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

        while anchor != last_anchor:
            label_window = uncertainty_labels[:,
                           anchor[0]:anchor[0] + window_size,
                           anchor[1]:anchor[1] + window_size
                           ]

            uncertainty_window = uncertainty_scores[:,
                                 anchor[0]:anchor[0] + window_size,
                                 anchor[1]:anchor[1] + window_size
                                 ]

            accuracy = torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)
            avg_uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

            accurate = accuracy < accuracy_threshold
            uncertain = avg_uncertainty >= uncertainty_threshold

            au += torch.sum(accurate & uncertain)
            ac += torch.sum(accurate & ~uncertain)
            iu += torch.sum(~accurate & uncertain)
            ic += torch.sum(~accurate & ~uncertain)

            if anchor[1] < uncertainty_labels.shape[1] - window_size:
                anchor = (anchor[0], anchor[1] + 1)
            else:
                anchor = (anchor[0] + 1, 0)

    a_given_c = ac / (ac + ic + 1e-10)
    u_given_i = iu / (ic + iu + 1e-10)

    pavpu = (ac + iu) / (ac + au + ic + iu + 1e-10)

    return pavpu, a_given_c, u_given_i


def roc_pr(uncertainty_scores, uncertainty_labels, window_size=1):
    if window_size == 1:
        y_true = uncertainty_labels.flatten().numpy()
        y_score = uncertainty_scores.flatten().numpy()
    else:
        y_true = []
        y_score = []

        anchor = (0, 0)
        last_anchor = (uncertainty_labels.shape[1] - window_size, uncertainty_labels.shape[2] - window_size)

        while anchor != last_anchor:
            label_window = uncertainty_labels[:,
                           anchor[0]:anchor[0] + window_size,
                           anchor[1]:anchor[1] + window_size
                           ]

            uncertainty_window = uncertainty_scores[:,
                                 anchor[0]:anchor[0] + window_size,
                                 anchor[1]:anchor[1] + window_size
                                 ]

            accuracy = (torch.sum(label_window, dim=(1, 2)) / (window_size ** 2)) > .5
            uncertainty = torch.mean(uncertainty_window, dim=(1, 2))

            for i in range(accuracy.shape[0]):
                y_true.append(accuracy[i].item())
                y_score.append(uncertainty[i].item())

            if anchor[1] < uncertainty_labels.shape[1] - window_size:
                anchor = (anchor[0], anchor[1] + 1)
            else:
                anchor = (anchor[0] + 1, 0)

        y_true = np.array(y_true)
        y_score = np.array(y_score)

    pr, rec, tr = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)

    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, ap, no_skill


def ece(y_pred, y_true, n_bins=10):
    y_true = y_true.long().argmax(dim=1)

    return torchmetrics.functional.calibration_error(
        y_pred,
        y_true,
        'multiclass',
        n_bins=n_bins,
        num_classes=y_pred.shape[1]
    )


def brier_score(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)


def tsne(y_pred, y_true, perplexity=10):
    y_true = y_true.argmax(dim=1)
    X_tsne = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity=perplexity).fit_transform(y_pred)

    plt.figure(figsize=(12, 8))
    colors = ['r', 'g', 'b', 'y']

    for i in range(4):
        mask = y_true == i

    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], label=f'Class {i}')
    plt.legend()
    plt.title('t-SNE Visualization of Semantic Segmentation Classes')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.show()
