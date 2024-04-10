# Trace it by
# * above sm80: nsys nvprof --print-gpu-trace python3 gemm_diff.py
# * below sm80: nvprof --print-gpu-trace python3 gemm_diff.py
# Different kernels are used depending on whether bias is given, causing precision error.

import torch
import torch.nn.functional as F

weight = torch.randn(6144, 2048, dtype=torch.bfloat16, device="cuda")
bias = torch.zeros(6144, dtype=torch.bfloat16, device="cuda")
x = torch.randn(2, 8, 2048, dtype=torch.bfloat16, device="cuda")

y1 = F.linear(x, weight)
y2 = F.linear(x, weight, bias)
print((y1 - y2).abs().max())
