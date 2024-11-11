import functools

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
from torch.utils.benchmark import Timer
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

# ======= MLP =======


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


def test_mlp(model_cls):
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
    torch.testing.assert_close(opt_model.fc1.weight.grad, ref_model.fc1.weight.grad)
    torch.testing.assert_close(opt_model.fc1.bias.grad, ref_model.fc1.bias.grad)
    torch.testing.assert_close(opt_model.fc2.weight.grad, ref_model.fc2.weight.grad)
    torch.testing.assert_close(opt_model.fc2.bias.grad, ref_model.fc2.bias.grad)


def perf_mlp(model_cls):
    model = model_cls(4096, 4096 * 4, "gelu_pytorch_tanh").cuda()

    x = torch.randn(3, 256, 4096, device="cuda", requires_grad=True)
    torch.cuda.empty_cache()
    prev_alloc = torch.cuda.memory_allocated()
    y = model(x)
    post_alloc = torch.cuda.memory_allocated()
    memory_cost = (post_alloc - prev_alloc) / (1 << 20)
    grad_y = torch.randn_like(y)

    fwd_cost = Timer("model(x)", globals=locals()).timeit(10).mean * 1e3
    bwd_cost = Timer("y.backward(grad_y, retain_graph=True)", globals=locals()).timeit(10).mean * 1e3

    return fwd_cost, bwd_cost, memory_cost


def with_checkpoint(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        return checkpoint(fn, *args, **kwargs)

    return wrap


test_mlp(FusedGeLUMLP)
test_mlp(RecomputeGeLUMLP)

ref_stats = perf_mlp(GeLUMLP)
fused_gelu_stats = perf_mlp(FusedGeLUMLP)
recomp_mlp_stats = perf_mlp(RecomputeGeLUMLP)

# gradient checkpoint
GeLUMLP.__call__ = with_checkpoint(GeLUMLP.__call__)
ckpt_stats = perf_mlp(GeLUMLP)


df = pd.DataFrame(
    [ref_stats, fused_gelu_stats, recomp_mlp_stats, ckpt_stats],
    columns=["fwd (ms)", "bwd (ms)", "memory (MB)"],
    index=["reference", "fused-gelu", "recompute", "grad-ckpt"],
)
print("======= MLP =======")
print(tabulate(df, headers=df.columns, tablefmt="psql", floatfmt=".2f"))


# ======= SwiGLU =======

from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP


class SiLUMulLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up, down_w)
        return F.linear(F.silu(gate) * up, down_w)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # grad_output: [b, s, h]
        gate, up, down_w = ctx.saved_tensors  # gate: [b, s, 4h], up: [b, s, 4h], down_w [h, 4h]

        gate_act = F.silu(gate)  # [b, s, 4h]
        gate_act_mul_up = gate_act * up  # [b, s, 4h]

        grad_down_w = grad_output.flatten(end_dim=-2).T @ gate_act_mul_up.flatten(
            end_dim=-2
        )  # [h, b*s] @ [b*s, 4h] = [h, 4h]
        grad_input = grad_output @ down_w  # [b, s, h] @ [h, 4h] = [b, s, 4h]

        grad_up = grad_input * gate_act
        grad_input = grad_input * up

        grad_gate = torch.ops.aten.silu_backward(grad_input, gate)

        return grad_gate, grad_up, grad_down_w


class FusedSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act in ["silu", "swish"], f"Activation function {config.hidden_act} not supported."

    def forward(self, x):
        return SiLUMulLinearFunction.apply(self.gate_proj(x), self.up_proj(x), self.down_proj.weight)


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, gate_w, up_w, down_w)
        return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # grad_output: [b, s, h]
        x, gate_w, up_w, down_w = ctx.saved_tensors  # gate: [b, s, 4h], up: [b, s, 4h], down_w [h, 4h]
        gate = F.linear(x, gate_w)  # [b, s, 4h]
        up = F.linear(x, up_w)  # [b, s, 4h]
        gate_act = F.silu(gate)  # [b, s, 4h]
        gate_act_up = gate_act * up  # [b, s, 4h]

        grad_down_w = grad_output.flatten(end_dim=-2).T @ gate_act_up.flatten(
            end_dim=-2
        )  # [h, b*s] @ [b*s, 4h] = [h, 4h]
        grad_gate_act_up = grad_output @ down_w  # [b, s, h] @ [h, 4h] = [b, s, 4h]

        grad_up = grad_gate_act_up * gate_act  # [b, s, 4h]
        grad_up_w = grad_up.flatten(end_dim=-2).T @ x.flatten(end_dim=-2)  # [4h, b*s] @ [b*s, h] -> [4h, h]

        grad_gate_act = grad_gate_act_up * up  # [b, s, 4h]
        grad_gate = torch.ops.aten.silu_backward(grad_gate_act, gate)  # [b, s, 4h]
        grad_gate_w = grad_gate.flatten(end_dim=-2).T @ x.flatten(end_dim=-2)  # [4h, b*s] @ [b*s, h] -> [4h, h]

        grad_x = grad_gate @ gate_w + grad_up @ up_w  # [b, s, h]
        return grad_x, grad_gate_w, grad_up_w, grad_down_w


class RecomputeSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act in ["silu", "swish"], f"Activation function {config.hidden_act} not supported."

    def forward(self, x):
        return SwiGLUFunction.apply(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)


def test_swiglu(model_cls):
    config = LlamaConfig()
    ref_model = LlamaMLP(config).cuda()
    opt_model = model_cls(config).cuda()
    opt_model.load_state_dict(ref_model.state_dict())

    ref_x = torch.randn(3, 17, config.hidden_size, device="cuda", requires_grad=True)
    opt_x = ref_x.detach().clone().requires_grad_()

    ref_y = ref_model(ref_x)
    opt_y = opt_model(opt_x)
    torch.testing.assert_close(opt_y, ref_y)

    grad_output = torch.randn_like(ref_y)
    ref_y.backward(grad_output, retain_graph=True)
    opt_y.backward(grad_output, retain_graph=True)
    torch.testing.assert_close(opt_x.grad, ref_x.grad)
    torch.testing.assert_close(opt_model.gate_proj.weight.grad, ref_model.gate_proj.weight.grad)
    torch.testing.assert_close(opt_model.up_proj.weight.grad, ref_model.up_proj.weight.grad)
    torch.testing.assert_close(opt_model.down_proj.weight.grad, ref_model.down_proj.weight.grad)


def perf_swiglu(model_cls):
    config = LlamaConfig()
    model = model_cls(config).cuda()

    x = torch.randn(3, 256, config.hidden_size, device="cuda", requires_grad=True)
    torch.cuda.empty_cache()
    prev_alloc = torch.cuda.memory_allocated()
    y = model(x)
    post_alloc = torch.cuda.memory_allocated()
    memory_cost = (post_alloc - prev_alloc) / (1 << 20)
    grad_y = torch.randn_like(y)

    fwd_cost = Timer("model(x)", globals=locals()).timeit(10).mean * 1e3
    bwd_cost = Timer("y.backward(grad_y, retain_graph=True)", globals=locals()).timeit(10).mean * 1e3

    return fwd_cost, bwd_cost, memory_cost


test_swiglu(FusedSwiGLU)
test_swiglu(RecomputeSwiGLU)

ref_stats = perf_swiglu(LlamaMLP)
fused_stats = perf_swiglu(FusedSwiGLU)
recomp_stats = perf_swiglu(RecomputeSwiGLU)

# gradient checkpoint
LlamaMLP.__call__ = with_checkpoint(LlamaMLP.__call__)
ckpt_stats = perf_swiglu(LlamaMLP)

df = pd.DataFrame(
    [ref_stats, fused_stats, recomp_stats, ckpt_stats],
    columns=["fwd (ms)", "bwd (ms)", "memory (MB)"],
    index=["reference", "fused-gelu", "recompute", "grad-ckpt"],
)
print("======= SwiGLU =======")
print(tabulate(df, headers=df.columns, tablefmt="psql", floatfmt=".2f"))
