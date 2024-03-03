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


@torch.compile
def rms_norm_triton(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = input.dtype
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    return weight * input.to(input_dtype)


def main():
    batch_size = 4
    eps = 1e-6
    timeit_number = 20

    dtype_choices = (torch.double, torch.float, torch.half, torch.bfloat16)
    seq_len_choices = (128, 512, 2048)
    hidden_size_choices = (32, 61, 64, 128, 130, 256, 512, 1024, 1027, 2048, 4096, 8192, 16384)

    # seq_len_choices = (2048,)
    # hidden_size_choices = (4096,)

    for dtype in dtype_choices:
        fwd_table = []
        bwd_table = []

        torch.cuda.empty_cache()
        for seq_len, hidden_size in tqdm(list(product(seq_len_choices, hidden_size_choices))):
            weight = (torch.randn(hidden_size, dtype=dtype, device="cuda") + 1).requires_grad_(True)
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, dtype=dtype, device="cuda", requires_grad=True
            )
            grad = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")

            # forward check
            naive_output = rms_norm_naive(hidden_states, weight, eps)

            ts_output = rms_norm_ts(hidden_states, weight, eps)
            torch.testing.assert_close(naive_output, ts_output)

            triton_output = rms_norm_triton(hidden_states, weight, eps)
            torch.testing.assert_close(naive_output, triton_output, rtol=1e-2, atol=1e-5)

            cuda_output = rms_norm_cuda(hidden_states, weight, eps)
            torch.testing.assert_close(naive_output, cuda_output, rtol=1e-2, atol=1e-5)

            # backward check
            naive_output.backward(grad, retain_graph=True)
            naive_dgrad, naive_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None

            ts_output.backward(grad, retain_graph=True)
            ts_dgrad, ts_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            torch.testing.assert_close(ts_dgrad, naive_dgrad, rtol=1e-3, atol=1e-1)
            torch.testing.assert_close(ts_wgrad, naive_wgrad, rtol=1e-3, atol=1e-2)

            triton_output.backward(grad, retain_graph=True)
            triton_dgrad, triton_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            torch.testing.assert_close(triton_dgrad, naive_dgrad, rtol=1e-2, atol=1e-1)
            torch.testing.assert_close(triton_wgrad, naive_wgrad, rtol=5e-2, atol=1.0)

            cuda_output.backward(grad, retain_graph=True)
            cuda_dgrad, cuda_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            torch.testing.assert_close(cuda_dgrad, naive_dgrad, rtol=1e-2, atol=1e-1)
            torch.testing.assert_close(cuda_wgrad, naive_wgrad, rtol=5e-2, atol=1.0)

            # forward benchmark
            naive_stats = Timer("rms_norm_naive(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(
                timeit_number
            )
            ts_stats = Timer("rms_norm_ts(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(
                timeit_number
            )
            triton_stats = Timer(
                "rms_norm_triton(hidden_states, weight, eps)", globals={**globals(), **locals()}
            ).timeit(timeit_number)
            cuda_stats = Timer("rms_norm_cuda(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(
                timeit_number
            )
            fwd_table.append(
                [
                    hidden_states.shape,
                    naive_stats.mean * 1e6,
                    ts_stats.mean * 1e6,
                    triton_stats.mean * 1e6,
                    cuda_stats.mean * 1e6,
                    triton_stats.mean / cuda_stats.mean,
                ]
            )

            # backward benchmark
            naive_stats = Timer(
                "naive_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}
            ).timeit(timeit_number)
            ts_stats = Timer("ts_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}).timeit(
                timeit_number
            )
            triton_stats = Timer(
                "triton_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}
            ).timeit(timeit_number)
            cuda_stats = Timer(
                "cuda_output.backward(grad, retain_graph=True)", globals={**globals(), **locals()}
            ).timeit(timeit_number)
            bwd_table.append(
                [
                    hidden_states.shape,
                    naive_stats.mean * 1e6,
                    ts_stats.mean * 1e6,
                    triton_stats.mean * 1e6,
                    cuda_stats.mean * 1e6,
                    triton_stats.mean / cuda_stats.mean,
                ]
            )

        print(f"{dtype} forward:")
        print(
            tabulate(
                fwd_table,
                headers=[
                    "shape",
                    "rms_norm_naive",
                    "rms_norm_ts",
                    "rms_norm_triton",
                    "rms_norm_cuda",
                    "speedup vs triton",
                ],
                tablefmt="psql",
                floatfmt=".3f",
            )
        )
        print(f"{dtype} backward:")
        print(
            tabulate(
                bwd_table,
                headers=[
                    "shape",
                    "rms_norm_naive",
                    "rms_norm_ts",
                    "rms_norm_triton",
                    "rms_norm_cuda",
                    "speedup vs triton",
                ],
                tablefmt="psql",
                floatfmt=".3f",
            ),
        )


if __name__ == "__main__":
    main()
