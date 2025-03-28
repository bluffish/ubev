import torch
import torch.nn as nn
from models.backbones.simplebev.segnet import Segnet

import numpy as np
import torch
import torch.nn.functional as F

import models.backbones.simplebev.utils.vox


class SimpleBEV(nn.Module):
    def __init__(
            self,
            n_classes=2,
            Z = 200,
            Y = 8,
            X = 200,
    ):
        super(SimpleBEV, self).__init__()

        self.Z = Z
        self.Y = Y
        self.X = X

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                      scene_centroid_y,
                                      scene_centroid_z]).reshape([1, 3])
        scene_centroid = torch.from_numpy(scene_centroid_py).float()

        XMIN, XMAX = -50, 50
        ZMIN, ZMAX = -50, 50
        YMIN, YMAX = -5, 5
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

        self.vox_util = models.backbones.simplebev.utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid,
            bounds=bounds,
            assert_cube=False)

        self.back = Segnet(Z, Y, X, self.vox_util, encoder_type="effb4")

    def forward(self, images, intrinsics, extrinsics):
        B = images.shape[0]
        S = images.shape[1]
        device = images.device

        rgb_camXs = images[:, [1, 0, 2, 3, 4, 5], ...]
        intrinsics = intrinsics[:, [1, 0, 2, 3, 4, 5], ...]
        extrinsics = extrinsics[:, [1, 0, 2, 3, 4, 5], ...]

        pix_T_cams = torch.zeros((B, S, 4, 4)).to(device)
        pix_T_cams[:, :, :3, :3] = intrinsics
        pix_T_cams[:, :, -1, -1] = 1.

        extrinsics = torch.linalg.inv(extrinsics)
        rots = torch.linalg.inv(extrinsics[:, :, :3, :3])
        trans = extrinsics[:, :, :3, 3]

        velo_T_cams = models.backbones.simplebev.utils.geom.merge_rtlist(rots, trans).to(device)
        cam0_T_camXs = models.backbones.simplebev.utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)

        return self.back(rgb_camXs, pix_T_cams, cam0_T_camXs, self.vox_util)