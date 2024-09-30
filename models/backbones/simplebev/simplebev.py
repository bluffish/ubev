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
            n_classes=4,
            Z = 200,
            Y = 8,
            X = 200,
    ):
        super(SimpleBEV, self).__init__()

        self.back = Segnet(Z=Z, Y=Y, X=X)
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

    def forward(self, images, intrinsics, extrinsics):
        return self.back(images, intrinsics, extrinsics, self.vox_util)