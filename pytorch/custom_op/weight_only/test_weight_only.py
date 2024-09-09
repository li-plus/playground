import torch
import torch.nn as nn
import weight_only
from torch.utils.benchmark import Timer


def quantize_8bit(weight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    q_weight = (
        weight.view(-1, group_size)
        .div(scales.view(-1, 1))
        .add(2**7)
        .round()
        .clamp(min=0, max=2**8 - 1)
        .to(torch.uint8)
        .view(-1, 4)
    )
    q_weight = torch.stack((q_weight[:, 0], q_weight[:, 2], q_weight[:, 1], q_weight[:, 3]), dim=-1).view(weight.shape)
    return q_weight


def dequantize_8bit(q_weight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    weight = q_weight.to(scales.dtype).sub(2**7).view(-1, group_size).mul(scales.view(-1, 1)).view(-1, 4)
    weight = torch.stack((weight[:, 0], weight[:, 2], weight[:, 1], weight[:, 3]), dim=-1).view(q_weight.shape)
    return weight


def pseudo_quantize(weight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    return (
        weight.view(-1, group_size)
        .div(scales.view(-1, 1))
        .round()
        .clamp(min=-(2**7), max=2**7 - 1)
        .mul(scales.view(-1, 1))
        .view(weight.shape)
    )


@torch.no_grad()
def test_gemv_w8():
    M = 4096 * 4
    N = 4096

    fc = nn.Linear(N, M, device="cuda", dtype=torch.half)

    group_size = 128
    scales = fc.weight.data.view(-1, 128).amax(dim=-1) / (2**7 - 1)

    fc.weight.data.copy_(pseudo_quantize(fc.weight.data, scales=scales, group_size=group_size))

    x = torch.randn(N, dtype=torch.half, device="cuda")
    q_weight = quantize_8bit(fc.weight.data, scales, group_size=group_size)
    dq_weight = dequantize_8bit(q_weight, scales, group_size=group_size)
    torch.testing.assert_close(dq_weight, fc.weight.data)

    y_ref = fc(x)
    y_out = torch.ops.weight_only.gemv_w8(x, q_weight, scales, fc.bias.data)

    torch.testing.assert_close(y_out, y_ref, rtol=1e-3, atol=2e-2)

    elapsed = (
        Timer(
            "fc(x)",
            globals={**globals(), **locals()},
        )
        .timeit(100)
        .mean
    )
    bandwidth = (fc.weight.nbytes + fc.bias.nbytes + x.nbytes + y_ref.nbytes) / 1e9 / elapsed
    print(f"[fp16] elapsed {elapsed * 1e6:.3f} us, bandwidth {bandwidth:.3f} GB/s")

    elapsed = (
        Timer(
            "torch.ops.weight_only.gemv_w8(x, q_weight, scales, fc.bias.data)",
            globals={**globals(), **locals()},
        )
        .timeit(100)
        .mean
    )
    bandwidth = (q_weight.nbytes + scales.nbytes + fc.bias.nbytes + x.nbytes + y_out.nbytes) / 1e9 / elapsed
    print(f"[8bit] elapsed {elapsed * 1e6:.3f} us, bandwidth {bandwidth:.3f} GB/s")


if __name__ == "__main__":
    test_gemv_w8()
