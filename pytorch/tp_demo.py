"""
Usage: torchrun --master_port=23456 --nproc_per_node=2 tp_demo.py
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, set_seed


class AllReduceBackwardOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.chunk(group.size(), dim=0)[group.rank()])
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.chunk(group.size(), dim=0)[group.rank()])
        self.group = group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = AllReduceBackwardOp.apply(input, self.group)
        return F.linear(input, self.weight, self.bias)


class AllReduceForwardOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
        dist.all_reduce(input, group=group)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output, None


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.chunk(group.size(), dim=1)[group.rank()])
        self.bias = linear.bias
        self.group = group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.linear(input, self.weight)
        output = AllReduceForwardOp.apply(output, self.group)
        if self.bias is not None:
            output = output + self.bias
        return output


def main():
    set_seed(12345)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))  # single node
    torch.cuda.set_device(local_rank)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    batch_size = 2
    seq_len = 1024
    max_grad_norm = 1e6

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    input_ids = torch.arange(0, batch_size * seq_len, device="cuda").view(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, device="cuda")

    # run single-gpu forward
    output_tp1 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    grad_output = torch.randn_like(output_tp1.logits) / 1e4
    output_tp1.logits.backward(grad_output)
    embed_grad_tp1 = model.model.embed_tokens.weight.grad
    grad_norm_tp1 = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).float()
    for param in model.parameters():
        param.grad = None

    # apply tensor parallel
    group = dist.group.WORLD
    for layer in model.model.layers:
        layer.self_attn.q_proj = ColumnParallelLinear(layer.self_attn.q_proj, group=group)
        layer.self_attn.k_proj = ColumnParallelLinear(layer.self_attn.k_proj, group=group)
        layer.self_attn.v_proj = ColumnParallelLinear(layer.self_attn.v_proj, group=group)
        layer.self_attn.o_proj = RowParallelLinear(layer.self_attn.o_proj, group=group)
        layer.mlp.gate_proj = ColumnParallelLinear(layer.mlp.gate_proj, group=group)
        layer.mlp.up_proj = ColumnParallelLinear(layer.mlp.up_proj, group=group)
        layer.mlp.down_proj = RowParallelLinear(layer.mlp.down_proj, group=group)
        # config patch
        layer.self_attn.num_heads //= world_size
        layer.self_attn.num_key_value_heads //= world_size

    output_tpn = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    output_tpn.logits.backward(grad_output)
    embed_grad_tpn = model.model.embed_tokens.weight.grad

    # compute grad_norm
    shard_grads = []
    for m in model.modules():
        if isinstance(m, ColumnParallelLinear):
            shard_grads.append(m.weight.grad)
            if m.bias is not None:
                shard_grads.append(m.bias.grad)
        elif isinstance(m, RowParallelLinear):
            shard_grads.append(m.weight.grad)

    full_grads = {x.grad for x in model.parameters()} - set(shard_grads)
    grad_sq = torch.cat([x.flatten() for x in shard_grads]).float().square().sum()
    dist.all_reduce(grad_sq)
    grad_sq += torch.cat([x.flatten() for x in full_grads]).float().square().sum()
    grad_norm_tpn = grad_sq.sqrt()

    torch.testing.assert_close(output_tpn, output_tp1, rtol=0.1, atol=1.0)
    torch.testing.assert_close(embed_grad_tpn, embed_grad_tp1, rtol=0.1, atol=1.0)
    torch.testing.assert_close(grad_norm_tpn, grad_norm_tp1, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    main()
