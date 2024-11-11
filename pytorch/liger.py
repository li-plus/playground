import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import apply_liger_kernel_to_llama
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama


def show_memory_usage(message: str):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{message}]: {allocated=:.2f} GB, {reserved=:.2f} GB")


input_ids = torch.randint(0, 10000, (8, 1024), device="cuda")
attention_mask = torch.ones_like(input_ids)


def run_model():
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    show_memory_usage("pre-forward")
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    show_memory_usage("post-forward")
    output.logits.sum().backward()
    show_memory_usage("post-backward")


print("===== native =====")
run_model()

print("===== liger =====")
apply_liger_kernel_to_llama()
run_model()

print("===== liger + grad_ckpt =====")


def with_checkpoint(fn):
    @functools.wraps(fn)
    def wrap(*args, **kwargs):
        return checkpoint(fn, *args, **kwargs)

    return wrap


modeling_llama.LlamaMLP.__call__ = with_checkpoint(modeling_llama.LlamaMLP.__call__)
run_model()

print("===== liger + silu-mul-linear fusion =====")


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


class FusedSwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return SiLUMulLinearFunction.apply(self.gate_proj(x), self.up_proj(x), self.down_proj.weight)


def test_silu_mul_linear_function():
    ref_gate = torch.randn(4, 1024, device="cuda", requires_grad=True)
    ref_up = torch.randn(4, 1024, device="cuda", requires_grad=True)
    ref_down_w = torch.randn(256, 1024, device="cuda", requires_grad=True)

    opt_gate = ref_gate.detach().clone().requires_grad_()
    opt_up = ref_up.detach().clone().requires_grad_()
    opt_down_w = ref_down_w.detach().clone().requires_grad_()

    ref_output = F.linear(F.silu(ref_gate) * ref_up, ref_down_w)
    opt_output = SiLUMulLinearFunction.apply(opt_gate, opt_up, opt_down_w)

    torch.testing.assert_close(opt_output, ref_output)

    grad_output = torch.randn_like(ref_output)

    ref_output.backward(grad_output)
    opt_output.backward(grad_output)

    torch.testing.assert_close(opt_gate.grad, ref_gate.grad)
    torch.testing.assert_close(opt_up.grad, ref_up.grad)
    torch.testing.assert_close(opt_down_w.grad, ref_down_w.grad)


test_silu_mul_linear_function()


modeling_llama.LlamaMLP = FusedSwiGLUMLP
run_model()
