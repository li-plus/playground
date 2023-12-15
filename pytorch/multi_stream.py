"""
Compute small GEMMs using single stream vs multiple streams.

nsys profile python3 multi_stream.py

<torch.utils.benchmark.utils.common.Measurement object at 0x7f184e933b20>
single_stream(matrices)
  11.59 ms
  1 measurement, 10 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7f18d4e67ac0>
multi_stream(matrices, streams)
  10.11 ms
  1 measurement, 10 runs , 1 thread
"""

import torch
from torch.utils.benchmark import Timer


def single_stream(matrices):
    for mat in matrices:
        mat @ mat.T


def multi_stream(matrices, streams):
    num_gemms_per_stream = (len(matrices) + len(streams)) // len(streams)
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            for i, mat in enumerate(matrices[i * num_gemms_per_stream : (i + 1) * num_gemms_per_stream]):
                mat @ mat.T


torch.cuda.set_device(0)
matrices = [torch.randn(2048, 128, device='cuda') for _ in range(128)]
streams = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(2)]

print(Timer(stmt='single_stream(matrices)', globals=globals()).timeit(10))
print(Timer(stmt='multi_stream(matrices, streams)', globals=globals()).timeit(10))
