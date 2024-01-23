from models.model import Model
from tools.loss import *
from tools.uncertainty import *


class Baseline(Model):
    def __init__(self, *args, **kwargs):
        super(Baseline, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits, mode='entropy'):
        if mode == 'aleatoric':
            soft = Baseline.activate(logits)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'entropy':
            return entropy(logits, dim=1)

    @staticmethod
    def epistemic(logits):
        return entropy(logits)

    @staticmethod
    def activate(logits):
        return torch.softmax(logits, dim=1)

    def loss(self, logits, target):
        if self.loss_type == 'ce':
            return ce_loss(logits, target, weights=self.weights).mean()
        elif self.loss_type == 'focal':
            return focal_loss(logits, target, weights=self.weights, n=self.gamma).mean()
        elif self.loss_type == 'al':
            return a_loss(logits, target, weights=self.weights).mean()
        else:
            raise NotImplementedError()

    def forward(self, images, intrinsics, extrinsics):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        return self.backbone(images, intrinsics, extrinsics)
