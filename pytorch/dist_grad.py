"""
usage: torchrun --nproc_per_node 8 dist_grad.py
Gradients are all reduced MEAN, not SUM
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.fc(x)


dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.cuda.set_device(rank)

m = Model().cuda()
m = DDP(m)

x = torch.full((1, 4), fill_value=rank + 1, dtype=torch.float, device="cuda")
y = m(x)
loss = y.sum()
loss.backward()

print(f"[RANK {rank}] weight.grad {m.module.fc.weight.grad}")
