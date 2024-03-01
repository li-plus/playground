from itertools import product

import torch
from rms_norm_cuda import rms_norm_cuda
from tabulate import tabulate
from torch.utils.benchmark import Timer
from tqdm import tqdm

# RMSNorm reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py


def rms_norm_naive(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


@torch.jit.script
def rms_norm_ts(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


@torch.compile(dynamic=False)
def rms_norm_triton(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


def main():
    batch_size = 4
    eps = 1e-6

    seq_len_choices = (128, 512, 2048)
    hidden_size_choices = (32, 61, 64, 128, 131, 256, 512, 1024, 1027, 2048, 4096)

    # seq_len_choices = (2048,)
    # hidden_size_choices = (4096,)

    fwd_table = []
    bwd_table = []

    for seq_len, hidden_size in tqdm(list(product(seq_len_choices, hidden_size_choices))):
        weight = (torch.randn(hidden_size, device="cuda") + 1).requires_grad_(True)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", requires_grad=True)
        grad = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        # forward check
        naive_output = rms_norm_naive(hidden_states, weight, eps)

        ts_output = rms_norm_ts(hidden_states, weight, eps)
        assert torch.allclose(naive_output, ts_output)

        triton_output = rms_norm_triton(hidden_states, weight, eps)
        assert torch.allclose(naive_output, triton_output)

        cuda_output = rms_norm_cuda(hidden_states, weight, eps)
        assert torch.allclose(naive_output, cuda_output)

        # backward check
        naive_output.backward(grad, retain_graph=True)
        naive_dgrad, naive_wgrad = hidden_states.grad, weight.grad
        hidden_states.grad, weight.grad = None, None

        ts_output.backward(grad, retain_graph=True)
        ts_dgrad, ts_wgrad = hidden_states.grad, weight.grad
        hidden_states.grad, weight.grad = None, None
        assert torch.allclose(ts_dgrad, naive_dgrad, atol=1e-5)
        assert torch.allclose(ts_wgrad, naive_wgrad, rtol=1e-3, atol=1e-3)

        triton_output.backward(grad, retain_graph=True)
        triton_dgrad, triton_wgrad = hidden_states.grad, weight.grad
        hidden_states.grad, weight.grad = None, None
        assert torch.allclose(triton_dgrad, naive_dgrad, atol=1e-5)
        assert torch.allclose(triton_wgrad, naive_wgrad, rtol=1e-3, atol=1e-3)

        cuda_output.backward(grad, retain_graph=True)
        cuda_dgrad, cuda_wgrad = hidden_states.grad, weight.grad
        hidden_states.grad, weight.grad = None, None
        assert torch.allclose(cuda_dgrad, naive_dgrad, atol=1e-5)
        assert torch.allclose(cuda_wgrad, naive_wgrad, rtol=1e-2, atol=1e-2)

        # forward benchmark
        naive_stats = Timer("rms_norm_naive(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(100)
        ts_stats = Timer("rms_norm_ts(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(100)
        triton_stats = Timer("rms_norm_triton(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(
            100
        )
        cuda_stats = Timer("rms_norm_cuda(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(100)
        fwd_table.append(
            [
                hidden_states.shape,
                naive_stats.mean * 1e6,
                ts_stats.mean * 1e6,
                triton_stats.mean * 1e6,
                cuda_stats.mean * 1e6,
            ]
        )

        # backward benchmark
        naive_stats = Timer("naive_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}).timeit(
            100
        )
        ts_stats = Timer("ts_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}).timeit(100)
        triton_stats = Timer(
            "triton_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}
        ).timeit(100)
        cuda_stats = Timer("cuda_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}).timeit(
            100
        )
        bwd_table.append(
            [
                hidden_states.shape,
                naive_stats.mean * 1e6,
                ts_stats.mean * 1e6,
                triton_stats.mean * 1e6,
                cuda_stats.mean * 1e6,
            ]
        )

    print("Forward:")
    print(
        tabulate(
            fwd_table,
            headers=["shape", "rms_norm_naive", "rms_norm_ts", "rms_norm_triton", "rms_norm_cuda"],
            tablefmt="psql",
            floatfmt=".3f",
        )
    )
    print("Backward:")
    print(
        tabulate(
            bwd_table,
            headers=["shape", "rms_norm_naive", "rms_norm_ts", "rms_norm_triton", "rms_norm_cuda"],
            tablefmt="psql",
            floatfmt=".3f",
        ),
    )


if __name__ == "__main__":
    main()
