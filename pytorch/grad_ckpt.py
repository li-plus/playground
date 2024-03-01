import torch
from deepspeed.runtime.utils import see_memory_usage
from torch.utils.checkpoint import checkpoint


def forward(x):
    y = torch.exp(x)
    z = torch.exp(y)
    return z


def baseline():
    torch.cuda.empty_cache()
    x = torch.randn(256, 1024, 1024, device="cuda", requires_grad=True)
    y = forward(x)
    see_memory_usage("w/o gradient checkpoint", force=True)


def grad_ckpt():
    x = torch.randn(256, 1024, 1024, device="cuda", requires_grad=True)
    y = checkpoint(forward, x, use_reentrant=True)
    see_memory_usage("w/ gradient checkpoint", force=True)


baseline()
grad_ckpt()
