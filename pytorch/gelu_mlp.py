import functools

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.utils.benchmark import Timer
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN


class GeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GeLULinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        return F.linear(F.gelu(x, approximate="tanh"), w, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, w = ctx.saved_tensors
        x_act = F.gelu(x, approximate="tanh")  # [b, s, 4h]
        grad_x_act = grad_output @ w  # [b, s, h] @ [h, 4h] -> [b, s, 4h]
        grad_w = grad_output.flatten(end_dim=-2).T @ x_act.flatten(end_dim=-2)  # [h, b*s] @ [b*s, 4h] -> [h, 4h]
        grad_b = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))  # [h]
        grad_x = torch.ops.aten.gelu_backward(grad_x_act, x, approximate="tanh")  # [b, s, 4h]
        return grad_x, grad_w, grad_b


class FusedGeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)
        assert "gelu" in hidden_act, f"not implemented for {hidden_act} activation"

    def forward(self, x) -> torch.Tensor:
        return GeLULinearFunction.apply(self.fc1(x), self.fc2.weight, self.fc2.bias)


class GeLUMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, fc1_w: torch.Tensor, fc1_b: torch.Tensor, fc2_w: torch.Tensor, fc2_b: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(x, fc1_w, fc1_b, fc2_w)
        x = F.linear(x, fc1_w, fc1_b)
        x = F.gelu(x, approximate="tanh")
        x = F.linear(x, fc2_w, fc2_b)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x, fc1_w, fc1_b, fc2_w = ctx.saved_tensors
        x_fc1 = F.linear(x, fc1_w, fc1_b)  # [b, s, 4h]
        x_act = F.gelu(x_fc1, approximate="tanh")  # [b, s, 4h]

        grad_fc2_w = grad_output.flatten(end_dim=-2).T @ x_act.flatten(end_dim=-2)  # [h, b*s] @ [b*s, 4h] -> [h, 4h]
        grad_fc2_b = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))  # [h]
        grad_x_act = grad_output @ fc2_w  # [b, s, h] @ [h, 4h] -> [b, s, 4h]

        grad_x_fc1 = torch.ops.aten.gelu_backward(grad_x_act, x_fc1, approximate="tanh")  # [b, s, 4h]

        grad_fc1_w = grad_x_fc1.flatten(end_dim=-2).T @ x.flatten(end_dim=-2)  # [4h, b*s] @ [b*s, h] -> [4h, h]
        grad_fc1_b = grad_x_fc1.sum(dim=tuple(range(grad_x_fc1.ndim - 1)))  # [4h]
        grad_x = grad_x_fc1 @ fc1_w  # [b, s, 4h] @ [4h, h] -> [b, s, h]

        return grad_x, grad_fc1_w, grad_fc1_b, grad_fc2_w, grad_fc2_b


class RecomputeGeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return GeLUMLPFunction.apply(x, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)


def test_gelu_linear_function():
    ref_x = torch.randn(3, 17, 4096, device="cuda", requires_grad=True)
    ref_w = torch.randn(1024, 4096, device="cuda", requires_grad=True)
    ref_b = torch.randn(1024, device="cuda", requires_grad=True)

    opt_x = ref_x.detach().clone().requires_grad_()
    opt_w = ref_w.detach().clone().requires_grad_()
    opt_b = ref_b.detach().clone().requires_grad_()

    ref_output = F.linear(F.gelu(ref_x, approximate="tanh"), ref_w, ref_b)
    opt_output = GeLULinearFunction.apply(opt_x, opt_w, opt_b)

    torch.testing.assert_close(opt_output, ref_output)

    grad_output = torch.randn_like(ref_output)

    ref_output.backward(grad_output)
    opt_output.backward(grad_output)

    torch.testing.assert_close(opt_x.grad, ref_x.grad)
    torch.testing.assert_close(opt_w.grad, ref_w.grad)
    torch.testing.assert_close(opt_b.grad, ref_b.grad)


def test_model(model_cls):
    ref_model = GeLUMLP(1024, 4096, "gelu_pytorch_tanh").cuda()
    opt_model = model_cls(1024, 4096, "gelu_pytorch_tanh").cuda()

    opt_model.load_state_dict(ref_model.state_dict())

    ref_x = torch.randn(3, 17, 1024, device="cuda", requires_grad=True)
    opt_x = ref_x.detach().clone().requires_grad_()

    ref_y = ref_model(ref_x)
    opt_y = opt_model(opt_x)
    torch.testing.assert_close(opt_y, ref_y)

    grad_output = torch.randn_like(ref_y)
    ref_y.backward(grad_output, retain_graph=True)
    opt_y.backward(grad_output, retain_graph=True)
    torch.testing.assert_close(opt_x.grad, ref_x.grad)


def perf_model(model_cls):
    model = model_cls(1024, 4096, "gelu_pytorch_tanh").cuda()

    x = torch.randn(3, 256, 1024, device="cuda", requires_grad=True)
    torch.cuda.empty_cache()
    prev_alloc = torch.cuda.memory_allocated()
    y = model(x)
    post_alloc = torch.cuda.memory_allocated()
    memory_cost = (post_alloc - prev_alloc) / (1 << 20)
    grad_y = torch.randn_like(y)

    fwd_cost = Timer("model(x)", globals=locals()).timeit(100).mean * 1e6
    bwd_cost = Timer("y.backward(grad_y, retain_graph=True)", globals=locals()).timeit(100).mean * 1e6

    return fwd_cost, bwd_cost, memory_cost


def with_checkpoint(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        return checkpoint(fn, *args, **kwargs)

    return wrap


test_gelu_linear_function()
test_model(FusedGeLUMLP)
test_model(RecomputeGeLUMLP)

ref_stats = perf_model(GeLUMLP)
fused_gelu_stats = perf_model(FusedGeLUMLP)
recomp_mlp_stats = perf_model(RecomputeGeLUMLP)

# gradient checkpoint
GeLUMLP.__call__ = with_checkpoint(GeLUMLP.__call__)
ckpt_stats = perf_model(GeLUMLP)


df = pd.DataFrame(
    [ref_stats, fused_gelu_stats, recomp_mlp_stats, ckpt_stats],
    columns=["fwd (us)", "bwd (us)", "memory (MB)"],
    index=["reference", "fused-gelu", "recompute", "grad-ckpt"],
)
print(tabulate(df, headers=df.columns, tablefmt="psql", floatfmt=".2f"))
