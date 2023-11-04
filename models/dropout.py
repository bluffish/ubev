from focal_loss.focal_loss import FocalLoss

from models.model import Model
from tools.loss import *
from tools.uncertainty import *


class Dropout(Model):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits):
        return entropy(logits)

    @staticmethod
    def epistemic(logits):
        var = torch.var(Dropout.activate(logits), dim=0)

        return 1 - 1 / var

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target):
        if self.loss_type == 'ce':
            return ce_loss(logits, target, weights=self.weights).mean()
        elif self.loss_type == 'focal':
            return focal_loss(logits, target, weights=self.weights, n=2).mean()
        elif self.loss_type == 'al':
            return a_loss(logits, target, weights=self.weights).mean()
        else:
            raise NotImplementedError()

    def forward(self, images, intrinsics, extrinsics):
        if self.training:
            return self.backbone(images, intrinsics, extrinsics)
        else:
            self.train()
            out = [self.backbone(images, intrinsics, extrinsics) for _ in range(20)]

            return torch.mean(torch.stack(out), dim=0)

