"""
Usage: torchrun --master_port=23456 --nproc_per_node=2 fsdp_ulysses.py
"""

import functools
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import transformers.models.llama.modeling_llama as modeling_llama
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, set_seed

logger = logging.getLogger(__name__)

ulysses_group: dist.ProcessGroup = None


class AllToAllOp(torch.autograd.Function):
    def forward(
        ctx, input: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        input_list = [x.contiguous() for x in input.chunk(group.size(), dim=scatter_dim)]
        output_list = [torch.empty_like(x) for x in input_list]
        dist.all_to_all(output_list, input_list, group=group)
        output = torch.cat(output_list, dim=gather_dim)
        return output

    def backward(ctx, grad_output: torch.Tensor):
        grad_output_list = [x.contiguous() for x in grad_output.chunk(ctx.group.size(), dim=ctx.gather_dim)]
        grad_input_list = [torch.empty_like(x) for x in grad_output_list]
        dist.all_to_all(grad_input_list, grad_output_list, group=ctx.group)
        grad_input = torch.cat(grad_input_list, dim=ctx.scatter_dim)
        return grad_input, None, None, None


def all_to_all_to_ulysses_region(
    input: torch.Tensor, scatter_dim: int, gather_dim: int, group: dist.ProcessGroup = None
):
    return AllToAllOp.apply(input, scatter_dim, gather_dim, group)


def apply_ulysses(_flash_attention_forward):
    @functools.wraps(_flash_attention_forward)
    def _flash_attention_forward_wrapper(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: bool = None,
    ):
        # scatter head & gather sequence
        query_states = all_to_all_to_ulysses_region(query_states, scatter_dim=2, gather_dim=1, group=ulysses_group)
        key_states = all_to_all_to_ulysses_region(key_states, scatter_dim=2, gather_dim=1, group=ulysses_group)
        value_states = all_to_all_to_ulysses_region(value_states, scatter_dim=2, gather_dim=1, group=ulysses_group)

        query_length *= ulysses_group.size()

        full_position_ids = torch.empty(
            (position_ids.shape[0], query_length), dtype=position_ids.dtype, device=position_ids.device
        )
        dist.all_gather_into_tensor(full_position_ids, position_ids)
        position_ids = full_position_ids

        attn_output = _flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            softcap=softcap,
            deterministic=deterministic,
        )

        # scatter sequence & gather head
        attn_output = all_to_all_to_ulysses_region(attn_output, scatter_dim=1, gather_dim=2, group=ulysses_group)
        return attn_output

    return _flash_attention_forward_wrapper


def main():
    global ulysses_group

    set_seed(12345)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))  # single node
    torch.cuda.set_device(local_rank)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    batch_size = 1
    seq_len = 8

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    input_ids = torch.arange(0, batch_size * seq_len, device="cuda").view(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, device="cuda")
    position_ids = torch.arange(0, seq_len, device="cuda").unsqueeze(0).expand(batch_size, seq_len)

    # fsdp forward
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=size_based_auto_wrap_policy,
        use_orig_params=True,
    )
    output_base = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)

    # fsdp backward
    grad_output = torch.randn_like(output_base.logits) / 1e4
    output_base.logits.backward(grad_output)
    with FSDP.summon_full_params(model, with_grads=True):
        embed_grad_base = model.model.embed_tokens.weight.grad
        for param in model.parameters():
            param.grad = None

    # apply ulysses
    modeling_llama._flash_attention_forward = apply_ulysses(modeling_llama._flash_attention_forward)

    # ulysses forward
    ulysses_group = dist.group.WORLD
    chunk_size = seq_len // ulysses_group.size()
    chunk_slice = slice(chunk_size * ulysses_group.rank(), chunk_size * (ulysses_group.rank() + 1))
    output_ulysses = model(
        input_ids=input_ids[:, chunk_slice],
        attention_mask=attention_mask[:, chunk_slice],
        position_ids=position_ids[:, chunk_slice],
        use_cache=False,
    )

    # ulysses backward
    output_ulysses.logits.backward(grad_output[:, chunk_slice])

    with FSDP.summon_full_params(model, with_grads=True):
        embed_grad_ulysses = model.model.embed_tokens.weight.grad

    torch.testing.assert_close(output_ulysses.logits, output_base.logits[:, chunk_slice], atol=1.0, rtol=1e-2)
    torch.testing.assert_close(embed_grad_base, embed_grad_ulysses, atol=1.0, rtol=1e-2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
