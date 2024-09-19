from models.model import Model
from tools.uncertainty import *
from tools.loss import *

class Dropout(Model):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits):
        return entropy(torch.mean(logits, dim=0), dim=1)

    @staticmethod
    def epistemic(logits):
        pred, _ = logits.max(dim=2)
        var = torch.var(pred, dim=0)
        return (1 - 1 / var).unsqueeze(1)

    @staticmethod
    def activate(logits):
        return torch.mean(torch.softmax(logits, dim=2), dim=0)

    def forward(self, images, intrinsics, extrinsics):
        self.train()
        out = [self.backbone(images, intrinsics, extrinsics) for _ in range(10)]

        return torch.stack(out)

    def loss(self, logits, target, reduction='mean'):
        target = target.repeat(logits.shape[0], 1, 1, 1)
        logits = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[2], logits.shape[3], logits.shape[4])

        if self.loss_type == 'ce':
            A = ce_loss(logits, target, weights=self.weights)
        elif self.loss_type == 'focal':
            A = focal_loss(logits, target, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        outs, preds, loss = self.train_step(images, intrinsics, extrinsics, labels)
        return outs, preds, loss, torch.tensor(0.0, dtype=loss.dtype, device=loss.device)

    def loss_ood(self, alpha, y, ood):
        return self.loss(alpha, y), torch.tensor(0.0, dtype=alpha.dtype, device=alpha.device)
