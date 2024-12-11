import torch

import ray


@ray.remote(num_gpus=1)
class GPUCounter:
    def __init__(self):
        self.i = torch.zeros(1, dtype=torch.long, device="cuda")

    def get(self):
        return self.i.item()

    def incr(self, value):
        self.i += value


c = GPUCounter.remote()

for _ in range(10):
    c.incr.remote(1)

print(ray.get(c.get.remote()))
