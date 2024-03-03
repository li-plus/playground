"""
Gradient Checkpoint: 
* forward: run forward function without grad
* backward: run forward with grad and then run backward
"""

import torch
from deepspeed.runtime.utils import see_memory_usage
from torch.utils.checkpoint import checkpoint


def forward(x):
    y = torch.exp(x)
    z = torch.exp(y)
    return z


def baseline(forward_fn):
    torch.cuda.empty_cache()
    x = torch.randn(256, 1024, 1024, device="cuda", requires_grad=True)
    y = forward_fn(x)
    see_memory_usage("w/o gradient checkpoint", force=True)


def grad_ckpt(forward_fn):
    torch.cuda.empty_cache()
    x = torch.randn(256, 1024, 1024, device="cuda", requires_grad=True)
    y = checkpoint(forward_fn, x, use_reentrant=True)
    see_memory_usage("w/ gradient checkpoint", force=True)


baseline(forward)
grad_ckpt(forward)


class CustomExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        grad_x = grad_y * y
        return grad_x


custom_exp = CustomExpFunction.apply


def custom_forward(x):
    y = custom_exp(x)
    z = custom_exp(y)
    return z


baseline(custom_forward)
grad_ckpt(custom_forward)
