import torch
from torch.utils.benchmark import Timer

a = torch.ones(1024, 1024, 1024, dtype=torch.float32, pin_memory=True)
elapsed = Timer(stmt="a.cuda(non_blocking=True)", globals=dict(a=a)).timeit(10).mean
bw = a.numel() * a.element_size() / 1e9 / elapsed
print(f"pinned memory d2h bandwidth: {bw:.3f} GB/s")

a = torch.ones(1024, 1024, 1024, dtype=torch.float32)
elapsed = Timer(stmt="a.cuda(non_blocking=True)", globals=dict(a=a)).timeit(10).mean
bw = a.numel() * a.element_size() / 1e9 / elapsed
print(f"non-pinned memory d2h bandwidth: {bw:.3f} GB/s")
