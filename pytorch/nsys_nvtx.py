"""
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsys_report -f true -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --cudabacktrace=kernel --python-backtrace=cuda --python-sampling=true \
    python3 nsys_nvtx.py
"""

import torch
from torchvision.models import resnet50

model = resnet50().cuda().eval()
x = torch.randn(2, 3, 1024, 1024, device="cuda")

for _ in range(10):
    with torch.cuda.nvtx.range("forward"):
        torch.cuda.synchronize()
        y = model(x)
        loss = y.sum()
        torch.cuda.synchronize()

    with torch.cuda.nvtx.range("backward"):
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
