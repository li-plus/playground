"""
From Online Softmax to FlashAttention: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
"""

import math

import torch
import torch.nn.functional as F


def safe_softmax(input: torch.Tensor) -> torch.Tensor:
    # 1st pass
    m = input.max()

    # 2nd pass
    s = torch.zeros(())
    for x in input:
        s += (x - m).exp()

    # 3rd pass
    output = torch.empty_like(input)
    for i, x in enumerate(input):
        output[i] = (x - m).exp() / s

    return output


def online_softmax(input: torch.Tensor) -> torch.Tensor:
    # 1st pass
    m = torch.tensor(float("-inf"))
    s = torch.zeros(())
    for x in input:
        m_old = m
        m = torch.max(m, x)
        s = s * (m_old - m).exp() + (x - m).exp()

    # 2nd pass
    output = torch.empty_like(input)
    for i, x in enumerate(input):
        output[i] = (x - m).exp() / s

    return output


def online_softmax_tile(input: torch.Tensor) -> torch.Tensor:
    tile_size = 4
    n = input.numel()
    assert n % tile_size == 0

    # 1st pass
    m = torch.tensor(float("-inf"))
    s = torch.zeros(())
    for start in range(0, n, tile_size):
        tile = input[start : start + tile_size]

        m_tile = tile.max()
        s_tile = (tile - m_tile).exp().sum()

        m_old = m
        m = torch.max(m, m_tile)

        s = s * (m_old - m).exp() + s_tile * (m_tile - m).exp()

    # 2nd pass
    output = torch.empty_like(input)
    for start in range(0, n, tile_size):
        output[start : start + tile_size] = (input[start : start + tile_size] - m).exp() / s

    return output


def check_online_softmax():
    input = torch.randn(1024)

    safe = safe_softmax(input)
    online = online_softmax(input)
    online_tile = online_softmax_tile(input)
    ref = input.softmax(dim=-1)

    torch.testing.assert_close(ref, safe)
    torch.testing.assert_close(ref, online)
    torch.testing.assert_close(ref, online_tile)


def flash_attn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    _, head_dim = query.shape
    scale = 1 / math.sqrt(head_dim)

    context = torch.empty_like(query)

    for i, q in enumerate(query):
        s = torch.zeros(())
        m = torch.tensor(float("-inf"))
        o = torch.zeros_like(value[0])
        for k, v in zip(key, value):
            x = q.dot(k) * scale

            m_old = m
            m = torch.max(m, x)

            s_old = s
            s = s * (m_old - m).exp() + (x - m).exp()

            o = (s_old / s) * (m_old - m).exp() * o + (x - m).exp() / s * v

        context[i] = o

    return context


def flash_attn_tile(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q_tile_size = 8
    kv_tile_size = 4
    seq_len, head_dim = query.shape
    assert seq_len % q_tile_size == 0 and seq_len % kv_tile_size == 0
    scale = 1 / math.sqrt(head_dim)

    context = torch.empty_like(query)

    for q_start in range(0, seq_len, q_tile_size):
        q = query[q_start : q_start + q_tile_size]

        s = torch.zeros(q_tile_size, 1)
        m = torch.full((q_tile_size, 1), fill_value=float("-inf"))
        o = torch.zeros(q_tile_size, head_dim)
        for kv_start in range(0, seq_len, kv_tile_size):
            k = key[kv_start : kv_start + kv_tile_size]
            v = value[kv_start : kv_start + kv_tile_size]

            x = (q @ k.T) * scale  # [q_tile, kv_tile]

            m_tile = x.max(dim=1, keepdim=True).values
            s_tile = (x - m_tile).exp().sum(dim=1, keepdim=True)
            o_tile = ((x - m_tile).exp() / s_tile) @ v

            m_old = m
            m = torch.max(m, m_tile)

            s_old = s
            s = s * (m_old - m).exp() + s_tile * (m_tile - m).exp()

            o = (s_old / s) * (m_old - m).exp() * o + (s_tile / s) * (m_tile - m).exp() * o_tile

        context[q_start : q_start + q_tile_size] = o

    return context


def flash_attn_2(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    _, head_dim = query.shape
    scale = 1 / math.sqrt(head_dim)

    context = torch.empty_like(query)

    for i, q in enumerate(query):
        s = torch.zeros(())
        m = torch.tensor(float("-inf"))
        o_s = torch.zeros_like(value[0])
        for k, v in zip(key, value):
            x = q.dot(k) * scale

            m_old = m
            m = torch.max(m, x)

            s = s * (m_old - m).exp() + (x - m).exp()

            # maintain the unscaled o_s, equal to o * s in flash 1
            o_s = (m_old - m).exp() * o_s + (x - m).exp() * v

        context[i] = o_s / s

    return context


def check_flash_attn():
    seq_len = 256
    head_dim = 32

    query = torch.randn(seq_len, head_dim)
    key = torch.randn(seq_len, head_dim)
    value = torch.randn(seq_len, head_dim)

    ref_context = F.scaled_dot_product_attention(query, key, value)

    flash_context = flash_attn(query, key, value)
    torch.testing.assert_close(ref_context, flash_context)

    flash_tile_context = flash_attn_tile(query, key, value)
    torch.testing.assert_close(ref_context, flash_tile_context)

    flash_2_context = flash_attn_2(query, key, value)
    torch.testing.assert_close(ref_context, flash_2_context)


check_online_softmax()
check_flash_attn()
