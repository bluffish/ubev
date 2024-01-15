from models.model import Model
from tools.loss import *
from tools.uncertainty import *


class Dropout(Model):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits):
        return entropy(torch.mean(logits, dim=0))

    @staticmethod
    def epistemic(logits):
        pred, _ = logits.max(dim=2)
        var = torch.var(pred, dim=0)
        return (1 - 1 / var)[:, None]

    @staticmethod
    def activate(logits):
        return torch.mean(torch.softmax(logits, dim=2), dim=0)

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
        print(self.training)
        if self.training:
            return self.backbone(images, intrinsics, extrinsics)
        else:
            self.train()
            out = [self.backbone(images, intrinsics, extrinsics) for _ in range(10)]

            return torch.stack(out)

