import torch
import torch.nn as nn

from models.backbones.fiery.fiery import Fiery
from models.backbones.cvt.cross_view_transformer import CrossViewTransformer
from models.backbones.lss.lift_splat_shoot import LiftSplatShoot

backbones = {
    'fiery': Fiery,
    'cvt': CrossViewTransformer,
    'lss': LiftSplatShoot
}


class Model(nn.Module):

    def __init__(self, devices, backbone='fiery', n_classes=4, opt=None, loss_type='focal'):
        super(Model, self).__init__()

        self.device = devices[0]

        self.weights = torch.tensor([3., 1., 2., 1.]).to(devices[0])

        self.backbone = nn.DataParallel(backbones[backbone](n_classes=n_classes).to(devices[0]),
                                        output_device=devices[0],
                                        device_ids=devices)

        self.loss_type = loss_type
        self.opt = opt

    @staticmethod
    def aleatoric(x): pass

    @staticmethod
    def epistemic(x): pass

    @staticmethod
    def activate(x): pass

    @staticmethod
    def loss(x, gt): pass

    def state_dict(self, epoch=-1):
        return {
            'model_state_dict': super().state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt is not None else None,
            'epoch': epoch
        }

    def load(self, state_dict):
        self.load_state_dict(state_dict['model_state_dict'])

        if self.opt is not None:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_step(self, images, intrinsics, extrinsics, labels):
        self.opt.zero_grad(set_to_none=True)

        outs = self(images, intrinsics, extrinsics)
        preds = self.activate(outs)

        loss = self.loss(outs, labels.to(self.device))
        loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        return outs, preds, loss

    def forward(self, images, intrinsics, extrinsics): pass
