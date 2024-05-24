"""
Compute small GEMMs using single stream vs multiple streams.

nsys profile python3 multi_stream.py

<torch.utils.benchmark.utils.common.Measurement object at 0x7f869b527e50>
single_stream(matrices)
  3.36 ms
  1 measurement, 10 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7f872e6cb490>
multi_stream(matrices, streams)
  2.22 ms
  1 measurement, 10 runs , 1 thread
"""

import torch
from torch.utils.benchmark import Timer


def single_stream(matrices):
    for mat in matrices:
        mat @ mat.T


def multi_stream(matrices, streams):
    for stream in streams:
        with torch.cuda.stream(stream):
            for i in range(0, len(matrices), len(streams)):
                mat = matrices[i]
                mat @ mat.T


torch.cuda.set_device(0)
matrices = [torch.randn(1024, 128, device="cuda") for _ in range(128)]
streams = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(2)]

print(Timer(stmt="single_stream(matrices)", globals=globals()).timeit(10))
print(Timer(stmt="multi_stream(matrices, streams)", globals=globals()).timeit(10))
