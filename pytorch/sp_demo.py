"""
Usage: torchrun --master_port=23456 --nproc_per_node=2 sp_demo.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, set_seed
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states = AllGatherOp.apply(hidden_states, self.group)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = ReduceScatterOp.apply(hidden_states, self.group)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = AllGatherOp.apply(hidden_states, self.group)
    hidden_states = self.mlp(hidden_states)
    hidden_states = ReduceScatterOp.apply(hidden_states, self.group)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


class AllGatherOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
        ctx.group = group
        output = torch.empty((input.shape[0] * group.size(), *input.shape[1:]), dtype=input.dtype, device=input.device)
        dist.all_gather_into_tensor(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = torch.empty(
            (grad_output.shape[0] // ctx.group.size(), *grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(grad_input, grad_output, group=ctx.group)
        return grad_input, None


class ReduceScatterOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
        ctx.group = group
        output = torch.empty((input.shape[0] // group.size(), *input.shape[1:]), dtype=input.dtype, device=input.device)
        dist.reduce_scatter_tensor(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        grad_input = torch.empty(
            (grad_output.shape[0] * ctx.group.size(), *grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.all_gather_into_tensor(grad_input, grad_output, group=ctx.group)
        return grad_input, None


class ColumnParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.weight = linear.weight.chunk(group.size(), dim=0)[group.rank()]
        self.bias = None
        if linear.bias is not None:
            self.bias = linear.bias.chunk(group.size(), dim=0)[group.rank()]
        self.group = group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group: dist.ProcessGroup) -> None:
        super().__init__()
        self.weight = linear.weight.chunk(group.size(), dim=1)[group.rank()]
        self.bias = linear.bias
        self.group = group

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias / self.group.size() if self.bias is not None else None
        return F.linear(input, self.weight, bias)


def main():
    set_seed(12345)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))  # single node
    torch.cuda.set_device(local_rank)

    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    batch_size = 8
    seq_len = 512

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
    for param in model.parameters():
        param.grad = None

    # apply tensor parallel & sequence parallel
    group = dist.group.WORLD
    for layer in model.model.layers:
        layer.group = group
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

    old_llama_decoder_layer_forward = LlamaDecoderLayer.forward
    LlamaDecoderLayer.forward = llama_decoder_layer_forward

    sp_size = input_ids.shape[0] // group.size()
    sp_slice = slice(sp_size * group.rank(), sp_size * (group.rank() + 1))
    output_tpn = model(input_ids=input_ids[sp_slice], attention_mask=attention_mask[sp_slice], use_cache=False)
    output_tpn.logits.backward(grad_output[sp_slice])
    embed_grad_tpn = model.model.embed_tokens.weight.grad
    dist.all_reduce(embed_grad_tpn, group=group)

    torch.testing.assert_close(output_tp1.logits[sp_slice], output_tpn.logits, rtol=0.1, atol=1.0)
    torch.testing.assert_close(embed_grad_tp1, embed_grad_tpn, rtol=0.1, atol=2.0)


if __name__ == "__main__":
    main()
