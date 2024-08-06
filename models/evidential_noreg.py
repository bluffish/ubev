from models.model import Model
from tools.loss import *
from tools.uncertainty import *
from tools.geometry import *


class EvidentialNoReg(Model):
    def __init__(self, *args, **kwargs):
        super(EvidentialNoReg, self).__init__(*args, **kwargs)

        # !! Will be overwrite after the model is created in train.py !!
        self.beta_lambda = 0.001
        self.ood_lambda = 0.01
        self.k = 64

    @staticmethod
    def aleatoric(alpha, mode='dissonance'):
        if mode == 'aleatoric':
            soft = EvidentialNoReg.activate(alpha)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'dissonance':
            return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, reduction='mean'):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def loss_ood(self, alpha, y, ood, K=40):
        A = self.loss(alpha, y, reduction='none')
        A *= 1 + (self.epistemic(alpha).detach() * self.k)

        oreg = ood_reg(alpha, ood) * self.ood_lambda
        A = A[~ood.bool()].mean()

        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        preds = self.activate(outs)
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

