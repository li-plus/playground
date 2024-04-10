# Reference: https://huggingface.co/docs/safetensors/index

import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.benchmark import Timer


def st_load(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def access(tensors):
    return sum(x.sum() for x in tensors.values())


tensors = {
    "weight1": torch.randn((1024, 1024, 256)),
    "weight2": torch.randn((1024, 1024, 256)),
}

with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
    # torch native
    pt_path = Path(d) / "model.pt"
    print(Timer("torch.save(tensors, pt_path)", globals=globals()).timeit(3))
    print(Timer("torch.load(pt_path)", globals=globals()).timeit(3))
    print(Timer("access(torch.load(pt_path))", globals=globals()).timeit(3))

    # safe tensors
    st_path = Path(d) / "model.safetensors"
    print(Timer("save_file(tensors, st_path)", globals=globals()).timeit(3))
    print(Timer("st_load(st_path)", globals=globals()).timeit(3))
    print(Timer("access(st_load(st_path))", globals=globals()).timeit(3))

    torch.testing.assert_close(access(torch.load(pt_path)), access(st_load(st_path)))
