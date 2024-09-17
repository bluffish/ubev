import torch.nn as nn
import hydra
import pyrootutils
import os

import torch
from einops import repeat


# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
pyrootutils.set_root('./models/backbones/pointbev', pythonpath=True)


class PointBEV(nn.Module):
    def __init__(
            self,
            n_classes=2,
    ):
        super(PointBEV, self).__init__()
        self.model = self.get_model()

    def get_model(self):
        with hydra.initialize(version_base="1.3", config_path="./configs"):
            cfg = hydra.compose(
                config_name="train.yaml",
                return_hydra_config=True,
                overrides=["model=PointBeV"],
            )

            cfg.paths.root_dir = './models/backbones/pointbev'

        model = hydra.utils.instantiate(cfg.model.net)
        return model

    def forward(self, images, intrinsics, extrinsics):
        rots = extrinsics[:, :, :3, :3]
        trans = extrinsics[:, :, :3, 3].unsqueeze(3)
        B = images.shape[0]
        bev_aug = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=1).to(images.device)
        egoTin_to_seq = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=1).to(images.device)

        images = images.unsqueeze(1)
        rots = rots.unsqueeze(1)
        trans = trans.unsqueeze(1)
        intrinsics = intrinsics.unsqueeze(1)

        out = self.model(images, rots, trans, intrinsics, bev_aug, egoTin_to_seq)

        binimg = out['bev']['binimg'].squeeze(1)
        mask = out['masks']['bev']['binimg'].squeeze(1)

        return binimg, mask
