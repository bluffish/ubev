import torch
import torch.distributions as D
import torch.nn.functional as F
from tools.uncertainty import *
import torch.nn as nn

from fvcore import nn as nnp


def ce_loss(logits, target, weights=None):
    return F.cross_entropy(logits, target, weight=weights, reduction='none')


def a_loss(logits, target, weights=None):
    ce = ce_loss(logits, target, weights=weights)
    al = entropy(logits)[:, 0, :, :].detach()

    return ce * al


def bce_loss(logits, target, weights=None):
    return F.binary_cross_entropy_with_logits(logits, target, reduction='none')


def focal_loss(logits, target, weights=None, n=2):
    target = target.argmax(dim=1)
    log_p = F.log_softmax(logits, dim=1)

    ce = F.nll_loss(log_p, target, weight=weights, reduction='none')
    log_pt = log_p.gather(1, target[None])

    pt = log_pt.exp()
    loss = ce * (1 - pt + 1e-8) ** n

    return loss


def sigmoid_focal_loss(logits, target, weights=None, n=2):
    return nnp.sigmoid_focal_loss(logits, target, gamma=n)


def uce_loss(alpha, y, weights=None):
    S = torch.sum(alpha, dim=1, keepdim=True)
    B = y * (torch.digamma(S) - torch.digamma(alpha) + 1e-10)

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def u_focal_loss(alpha, y, weights=None, n=2):
    S = torch.sum(alpha, dim=1, keepdim=True)

    a0 = S
    aj = torch.gather(alpha, 1, torch.argmax(y, dim=1, keepdim=True))

    B = y * torch.exp(
        (torch.lgamma(a0 - aj + n) + torch.lgamma(a0)) - (torch.lgamma(a0 + n) + torch.lgamma(a0 - aj))
    ) * (torch.digamma(a0 + n) - torch.digamma(aj))

    if weights is not None:
        B *= weights.view(1, -1, 1, 1)

    A = torch.sum(B, dim=1, keepdim=True)

    return A


def entropy_reg(alpha, beta_reg=.0005):
    alpha = alpha.permute(0, 2, 3, 1)

    reg = D.Dirichlet(alpha).entropy().unsqueeze(1)

    return -beta_reg * reg


def ood_reg(alpha, ood):
    alpha = alpha.permute(0, 2, 3, 1)

    alpha_d = D.Dirichlet(alpha)
    target_d = D.Dirichlet(torch.ones_like(alpha))

    reg = D.kl.kl_divergence(alpha_d, target_d).unsqueeze(1)

    return reg[ood.unsqueeze(1).bool()].mean()


def gamma(x):
    return torch.exp(torch.lgamma(x))