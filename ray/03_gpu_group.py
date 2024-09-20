"""
Adapted from:
https://github.com/ray-project/ray/blob/master/python/ray/train/torch/config.py
https://github.com/ray-project/ray/blob/master/python/ray/train/_internal/worker_group.py
"""

import os

import torch
import torch.distributed as dist

import ray


@ray.remote(num_cpus=8, num_gpus=1)
class TorchWorker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.getenv("CUDA_VISIBLE_DEVICES"))
        assert self.rank == self.local_rank, f"{self.rank=} {self.local_rank=}"
        self.data = torch.full((1024, 1024, 1024), fill_value=rank, dtype=torch.float32, device="cuda")

    def init_process_group(self):
        dist.init_process_group("nccl", world_size=self.world_size, rank=self.rank)

    def set_master(self, addr, port):
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

    @staticmethod
    def get_address_and_port():
        # See https://github.com/ray-project/ray/blob/master/python/ray/train/_internal/utils.py
        addr = ray.util.get_node_ip_address()
        port = 23456
        return addr, port

    def all_reduce(self):
        dist.all_reduce(self.data)
        return self.data.mean().item()


class TorchWorkerGroup:
    def __init__(self, world_size) -> None:
        self.workers = [TorchWorker.remote(rank, world_size) for rank in range(world_size)]

        addr, port = ray.get(self.workers[0].get_address_and_port.remote())

        for worker in self.workers:
            worker.set_master.remote(addr, port)
            worker.init_process_group.remote()

    def all_reduce(self):
        return ray.get([worker.all_reduce.remote() for worker in self.workers])


ray.init(num_cpus=65, num_gpus=8)
worker_group = TorchWorkerGroup(world_size=8)
print(worker_group.all_reduce())
