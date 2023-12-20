import torch

from models.model import Model
from tools.loss import *
from tools.uncertainty import *
from tools.geometry import *

import cv2
import matplotlib.pyplot as plt


class Evidential(Model):
    def __init__(self, *args, **kwargs):
        super(Evidential, self).__init__(*args, **kwargs)

        self.beta_lambda = 0.001
        self.ood_lambda = 0.1
        self.scale = 'none'
        self.k = 4

        print(f"BETA LAMBDA: {self.beta_lambda}")

        if self.loss_type == 'ce' and self.beta_lambda == 0:
            print("WARNING: USING UCE AND NO ENTROPY REG. WILL SET LAMBDA TO 0.001")
            self.beta_lambda = .001
        elif self.loss_type == 'focal' and self.beta_lambda > 0:
            print("WARNING: USING UFOCAL + ENTROPY REG. WILL SET LAMBDA TO 0")
            self.beta_lambda = .0

    @staticmethod
    def aleatoric(alpha, mode='var'):
        if mode == 'aleatoric':
            soft = Evidential.activate(alpha)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft[:, None, :, :]) / torch.max(1 - max_soft[:, None, :, :])
        elif mode == 'dissonance':
            return dissonance(alpha)
        elif mode == 'var':
            alpha0 = torch.sum(alpha, dim=1, keepdim=True)
            out = (alpha * (alpha0 - alpha)) / ((alpha0 * alpha0) * (alpha0 + 1))
            out = out.sum(dim=1, keepdim=True)
            return out / out.max()

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if self.scale == 'vac':
            scf = 1 + (self.epistemic(alpha).detach() * self.k)
            A *= scf

        if self.beta_lambda > 0:
            A += entropy_reg(alpha, self.beta_lambda)

        return A.mean()

    def loss_ood(self, alpha, y, ood):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if self.beta_lambda > 0:
            A += entropy_reg(alpha, beta_reg=self.beta_lambda)

        if self.scale == 'vac':
            scf = 1 + (self.epistemic(alpha).detach() * self.k)
            A *= scf

        oreg = ood_reg(alpha, ood) * self.ood_lambda

        A = A[(1 - ood).unsqueeze(1).bool()].mean()

        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        preds = self.activate(outs)

        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)
        loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics, limit=None):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

