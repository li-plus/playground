import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)

sub_group_ranks = [0, 1]
sub_group = dist.new_group(ranks=sub_group_ranks)

# allreduce
x = torch.full((1,), fill_value=rank + 1, dtype=torch.float32, device="cuda")
print(f"[{rank=}] before all_reduce: {x=}")
dist.all_reduce(x)
print(f"[{rank=}] after all_reduce: {x=}")

# reduce scatter
x = torch.full((8,), fill_value=rank + 1, dtype=torch.float32, device="cuda")
y = x[rank : rank + 1]
print(f"[{rank=}] before reduce_scatter: {x=}")
dist.reduce_scatter_tensor(y, x)
print(f"[{rank=}] after reduce_scatter: {x=}")
