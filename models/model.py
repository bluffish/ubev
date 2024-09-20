import torch
import torch.nn as nn

from models.backbones.cvt.cross_view_transformer import CrossViewTransformer
from models.backbones.fiery.fiery import Fiery
from models.backbones.lss.lift_splat_shoot import LiftSplatShoot
from models.backbones.pointbev.pointbev_model import PointBEV
from models.backbones.pointbev.bevformer_model import BEVFormer

backbones = {
    'fiery': Fiery,
    'cvt': CrossViewTransformer,
    'lss': LiftSplatShoot,
    'pointbev': PointBEV,
    'bevformer': BEVFormer,
}


class Model(nn.Module):
    def __init__(self, devices, backbone='fiery', n_classes=4, opt=None, scaler=None, loss_type='ce', weights=None):
        super(Model, self).__init__()

        self.device = devices[0]
        self.devices = devices

        self.weights = weights

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        self.bbt = backbone
        self.backbone = None

        self.loss_type = loss_type
        self.n_classes = n_classes
        self.opt = opt
        self.scaler = scaler
        self.gamma = .1
        self.tsne = False

        self.create_backbone(backbone)

    def create_backbone(self, backbone):
            self.backbone = nn.DataParallel(
                backbones[backbone](n_classes=self.n_classes).to(self.device),
                output_device=self.device,
                device_ids=self.devices,
            )

    @staticmethod
    def aleatoric(x):
        pass

    @staticmethod
    def epistemic(x):
        pass

    @staticmethod
    def activate(x):
        pass

    @staticmethod
    def loss(x, gt):
        pass

    def state_dict(self, epoch=-1):
        return {
            'model_state_dict': super().state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt is not None else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'epoch': epoch
        }

    def load(self, state_dict):
        self.load_state_dict(state_dict['model_state_dict'])

        if self.opt is not None:
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])

        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_step(self, images, intrinsics, extrinsics, labels):
        self.opt.zero_grad(set_to_none=True)

        if self.bbt == 'pointbev':
            outs, mask = self(images, intrinsics, extrinsics)
            loss = self.loss(outs, labels.to(self.device), reduction='none')

            loss = (loss.unsqueeze(1) * mask).sum() / (mask.sum() + 1e-6)
        else:
            outs = self(images, intrinsics, extrinsics)
            loss = self.loss(outs, labels.to(self.device))

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss

    def forward(self, images, intrinsics, extrinsics):
        pass
