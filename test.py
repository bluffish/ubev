import time

import numpy as np
import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(torch.__version__)
import torch.nn as nn


class Attn(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_attn = nn.Linear(embed_dim, 16 * embed_dim)

    def forward(self, x):
        k = self.c_attn(x)
        return k


yy = Attn(4096).to(torch.float32).cuda()
inp = torch.randn((2048, 4096)).cuda()
# yy_opt = torch.compile(yy, mode="reduce-overhead")
yy_opt = yy
sm = 0

for i in range(550):
    start = time.time()
    z = yy_opt(inp)
    end = time.time()

    # print(i, ' forward ', (end - start))

    sm += end - start

    z2 = torch.sum(z)

    start = time.time()
    z2.backward()
    end = time.time()

    # print(i, ' back ', (end - start))
    sm += end - start

print(i, ' Total: ', sm)