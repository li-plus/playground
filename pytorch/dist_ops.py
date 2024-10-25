# torchrun --nproc_per_node=4 dist_ops.py

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

# all_reduce
input = torch.full((1,), fill_value=rank, dtype=torch.float32, device="cuda")
print(f"[{rank=}] all_reduce: {input=}")
dist.all_reduce(input)
output = input
print(f"[{rank=}] all_reduce: {output=}")

# reduce_scatter
input = torch.arange(0, world_size, dtype=torch.float32, device="cuda")
output = input[rank : rank + 1]
print(f"[{rank=}] reduce_scatter: {input=}")
dist.reduce_scatter_tensor(output, input)
print(f"[{rank=}] reduce_scatter: {output=}")

# all_gather
input = torch.full((1,), fill_value=rank, dtype=torch.float32, device="cuda")
output = torch.empty(world_size, dtype=torch.float32, device="cuda")
print(f"[{rank=}] all_gather: {input=}")
dist.all_gather_into_tensor(output, input)
print(f"[{rank=}] all_gather: {output=}")

# all_to_all
input = torch.arange(0, world_size, dtype=torch.float32, device="cuda")
output = torch.empty_like(input)
print(f"[{rank=}] all_to_all: {input=}")
dist.all_to_all_single(output, input)
print(f"[{rank=}] all_to_all: {output=}")
