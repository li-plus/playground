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


tensors = {
    "weight1": torch.zeros((1024, 1024, 256)),
    "weight2": torch.zeros((1024, 1024, 256)),
}

with tempfile.TemporaryDirectory(prefix="/dev/shm/") as d:
    save_path = Path(d) / "model_ckpt"

    # torch native
    print(Timer("torch.save(tensors, save_path)", globals=globals()).timeit(3))
    print(Timer("torch.load(save_path)", globals=globals()).timeit(3))

    # safe tensors
    print(Timer("save_file(tensors, save_path)", globals=globals()).timeit(3))
    print(Timer("st_load(save_path)", globals=globals()).timeit(3))
