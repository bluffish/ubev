import cv2
import numpy as np
import torch
from sklearn.metrics import *
from sklearn.calibration import *
import torchmetrics


def get_iou(preds, labels, exclude=None):
    classes = preds.shape[1]
    iou = [0] * classes

    pmax = preds.argmax(dim=1)
    lmax = labels.argmax(dim=1)

    with torch.no_grad():
        for i in range(classes):
            p = (pmax == i).bool()
            l = (lmax == i).bool()

            if exclude is not None:
                p &= ~exclude
                l &= ~exclude

            intersect = (p & l).sum().float().item()
            union = (p | l).sum().float().item()
            iou[i] = intersect / union if union > 0 else 0
    return iou


def unc_iou(y_score, y_true, thresh=.5):
    pred = (y_score > thresh).bool()
    target = y_true.bool()

    intersect = (pred & target).sum()
    union = (pred | target).sum()

    return intersect / union


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


def roc_pr(uncertainty_scores, uncertainty_labels, exclude=None):

    y_true = uncertainty_labels.flatten().numpy()
    y_score = uncertainty_scores.flatten().numpy()

    if exclude is not None:
        include = ~exclude.flatten().numpy()
        y_true = y_true[include]
        y_score = y_score[include]

    pr, rec, tr = precision_recall_curve(y_true, y_score, drop_intermediate=True)
    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=True)

    aupr = auc(rec, pr)
    auroc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_score)

    no_skill = np.sum(y_true) / len(y_true)

    return fpr, tpr, rec, pr, auroc, ap, no_skill


def ece(y_pred, y_true, n_bins=10, exclude=None):
    y_true = y_true.long().argmax(dim=1)

    if exclude is not None:
        y_true[exclude] = -1

    return torchmetrics.functional.calibration_error(
        y_pred,
        y_true,
        'multiclass',
        n_bins=n_bins,
        num_classes=y_pred.shape[1],
        ignore_index=-1
    )


def brier_score(y_pred, y_true, exclude=None):
    brier = torch.nn.functional.mse_loss(y_pred, y_true, reduction='none')

    if exclude is not None:
        brier = brier[~exclude.unsqueeze(1).repeat(1, y_pred.shape[1], 1, 1)]

    return brier.mean()


def plot_acc_calibration(ax, mask, pred, labels, title='Calibration Plot', n_bins=20):
    mask = mask.permute(0, 2, 3, 1).reshape(-1)
    pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    labels = labels.permute(0, 2, 3, 1).reshape(-1, pred.shape[1]).argmax(dim=-1)

    pred_label = torch.max(pred[mask], 1)[1]
    p_value = torch.max(pred[mask], 1)[0]
    ground_truth = labels[mask]

    intervals = (p_value * n_bins).to(torch.int64).clamp(0, n_bins - 1)

    confidence_all = torch.bincount(intervals, minlength=n_bins).numpy()
    confidence_acc = torch.bincount(intervals, weights=(pred_label == ground_truth).to(torch.float32),
                                    minlength=n_bins).numpy()

    confidence_acc[confidence_all > 0] /= confidence_all[confidence_all > 0]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.bar(bin_centers, confidence_acc, alpha=0.7, width=bin_edges[1] - bin_edges[0], color='dodgerblue',
           label=f'Outputs')
    ax.plot([0, 1], [0, 1], ls='--', c='k')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
