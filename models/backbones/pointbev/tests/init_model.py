import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import hydra
import pyrootutils
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profiler import PyTorchProfiler
from torch.profiler import ProfilerActivity
from einops import repeat

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
device='cuda'
NUM_REPEAT = 1
B, Tin, N, C, H, W = 1, 1, 6, 3, 224, 480
Nq = 1

def get_model():
    with hydra.initialize(version_base="1.3", config_path="../configs"):
        cfg = hydra.compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model=BeVFormer"],
        )

        cfg.paths.root_dir = str(pyrootutils.find_root(indicator=".project-root"))

    model = hydra.utils.instantiate(cfg.model.net)
    return model


imgs = torch.randn(B, Tin, N, C, H, W).to(device)
rots = torch.randn(B, Tin, N, 3, 3).to(device)
trans = torch.randn(B, Tin, N, 3, 1).to(device)
intrins = torch.randn(B, Tin, N, 3, 3).to(device)
bev_aug = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=Tin).to(device)
egoTin_to_seq = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=Tin).to(device)

model = get_model().to(device)

out = model(imgs, rots, trans, intrins, bev_aug, egoTin_to_seq)

print(out['bev']['binimg'].shape)
# print(out['masks']['bev']['binimg'].shape)

cv2.imwrite("test.png", out['masks']['bev']['binimg'][0, 0, 0].long().cpu().numpy() * 255)