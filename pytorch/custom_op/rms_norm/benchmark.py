import torch
from rms_norm_cuda import rms_norm_cuda
from tabulate import tabulate
from torch.utils.benchmark import Timer

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
    batch_size = 64
    eps = 1e-6

    seq_len_choices = (128, 512, 2048)
    hidden_size_choices = (32, 61, 64, 128, 131, 256, 512, 1024, 1027, 2048, 4096)

    # seq_len_choices = (128,)
    # hidden_size_choices = (4096,)

    for seq_len in seq_len_choices:
        for hidden_size in hidden_size_choices:
            weight = (torch.randn(hidden_size, device="cuda") + 1).requires_grad_(True)
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", requires_grad=True)

            # forward check
            naive_output = rms_norm_naive(hidden_states, weight, eps)

            ts_output = rms_norm_ts(hidden_states, weight, eps)
            assert torch.allclose(naive_output, ts_output)

            triton_output = rms_norm_triton(hidden_states, weight, eps)
            assert torch.allclose(naive_output, triton_output)

            cuda_output = rms_norm_cuda(hidden_states, weight, eps)
            assert torch.allclose(naive_output, cuda_output)

            # backward check
            naive_loss = naive_output.sum().mul(2)
            naive_loss.backward(retain_graph=True)
            naive_dgrad, naive_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None

            ts_loss = ts_output.sum().mul(2)
            ts_loss.backward(retain_graph=True)
            ts_dgrad, ts_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            assert torch.allclose(ts_dgrad, naive_dgrad, atol=1e-5)
            assert torch.allclose(ts_wgrad, naive_wgrad, rtol=1e-3, atol=1e-3)

            triton_loss = triton_output.sum().mul(2)
            triton_loss.backward(retain_graph=True)
            triton_dgrad, triton_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            assert torch.allclose(triton_dgrad, naive_dgrad, atol=1e-5)
            assert torch.allclose(triton_wgrad, naive_wgrad, rtol=1e-3, atol=1e-3)

            cuda_loss = cuda_output.sum().mul(2)
            cuda_loss.backward(retain_graph=True)
            cuda_dgrad, cuda_wgrad = hidden_states.grad, weight.grad
            hidden_states.grad, weight.grad = None, None
            assert torch.allclose(cuda_dgrad, naive_dgrad, atol=1e-5)
            assert torch.allclose(cuda_wgrad, naive_wgrad, rtol=1e-2, atol=1e-2)

            # forward benchmark
            table = []

            stats = Timer("rms_norm_naive(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_naive", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("rms_norm_ts(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_ts", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("rms_norm_triton(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_triton", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("rms_norm_cuda(hidden_states, weight, eps)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_cuda", f"{stats.mean*1e3:.3f} ms"])

            # backward benchmark
            stats = Timer("naive_loss.backward(retain_graph=True)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_naive_backward", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("ts_loss.backward(retain_graph=True)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_ts_backward", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("triton_loss.backward(retain_graph=True)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_triton_backward", f"{stats.mean*1e3:.3f} ms"])

            stats = Timer("cuda_loss.backward(retain_graph=True)", globals={**globals(), **locals()}).timeit(10)
            table.append([hidden_states.shape, "rms_norm_cuda_backward", f"{stats.mean*1e3:.3f} ms"])

            print(tabulate(table, headers=["shape", "impl", "time"], tablefmt="psql"))


if __name__ == "__main__":
    main()
