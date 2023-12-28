"""
usage:
torchrun --nproc_per_node 8 comm_overlap.py

open profiler logs with: https://ui.perfetto.dev/
"""

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

# non overlap

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=4, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"log/non_overlap/rank_{rank}", use_gzip=True),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as p:
    a = torch.ones(1024 // world_size, 1024, dtype=torch.float32, device="cuda")
    b = torch.empty(1024, 1024, dtype=torch.float32, device="cuda")
    for step in range(8):
        for mini_step in range(16):
            dist.all_gather_into_tensor(b, a)
            b @ b
        torch.cuda.synchronize()
        p.step()

# overlap

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=4, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"log/overlap/rank_{rank}", use_gzip=True),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
) as p:
    a = torch.ones(1024 // world_size, 1024, dtype=torch.float32, device="cuda")
    b = torch.empty(1024, 1024, dtype=torch.float32, device="cuda")
    next_b = torch.empty_like(b)
    for step in range(8):
        handle = None
        next_handle = dist.all_gather_into_tensor(next_b, a, async_op=True)
        for mini_step in range(16):
            b, next_b = next_b, b
            handle, next_handle = next_handle, handle
            if mini_step < 16 - 1:
                next_handle = dist.all_gather_into_tensor(next_b, a, async_op=True)
            handle.wait()  # insert a sync event in the main cuda stream
            b @ b
        torch.cuda.synchronize()
        p.step()
