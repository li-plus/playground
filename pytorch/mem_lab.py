import os

import torch
import torch.nn as nn
from pytorch_memlab import profile, set_target_gpu

local_rank = int(os.getenv("LOCAL_RANK", "0"))
set_target_gpu(local_rank)


@profile
def func():
    x = torch.randn(4 * 1024, 1024, device="cuda")
    model = nn.Linear(1024, 1024 * 4, bias=False, device="cuda")
    y = model(x)
    z = y.mean()
    return z


func()
