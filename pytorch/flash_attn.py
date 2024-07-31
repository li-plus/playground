"""
From Online Softmax to FlashAttention: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
"""

import math

import torch
import torch.nn.functional as F


def safe_softmax(input: list[float]) -> list[float]:
    # 1st pass
    m = max(input)

    # 2nd pass
    s = 0.0
    for x in input:
        s += math.exp(x - m)

    # 3rd pass
    output = [math.exp(x - m) / s for x in input]

    return output


def online_softmax(input: list[float]) -> list[float]:
    # 1st pass
    m = float("-inf")
    s = 0.0
    for x in input:
        m_old = m
        m = max(m, x)
        s = s * math.exp(m_old - m) + math.exp(x - m)

    # 2nd pass
    output = [math.exp(x - m) / s for x in input]

    return output


def check_online_softmax():
    input = torch.randn(1024)

    safe = torch.tensor(safe_softmax(input.tolist()))
    online = torch.tensor(online_softmax(input.tolist()))
    ref = input.softmax(dim=-1)

    torch.testing.assert_close(ref, safe)
    torch.testing.assert_close(ref, online)


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


def check_flash_attn():
    seq_len = 256
    head_dim = 32

    query = torch.randn(seq_len, head_dim)
    key = torch.randn(seq_len, head_dim)
    value = torch.randn(seq_len, head_dim)

    ref_context = F.scaled_dot_product_attention(query, key, value)
    flash_context = flash_attn(query, key, value)
    torch.testing.assert_close(ref_context, flash_context)


check_online_softmax()
check_flash_attn()
