import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
from torch.utils.benchmark import Timer
import gemv_w3


def quantize_3bit(weight: torch.Tensor, scales: torch.Tensor, group_size:int=128) -> torch.Tensor:
    _, n = weight.shape
    assert n % group_size == 0
    q_weight = (weight.view(-1, group_size) / scales.view(-1, 1)).round().clamp(min=-4, max=3).to(torch.uint8)
    print(f'{q_weight=}')

    q_weight_lo = q_weight.view(-1, 4) & 0x3
    q_weight_lo = (q_weight_lo[:, 0] << 0) | (q_weight_lo[:, 1] << 2) | (q_weight_lo[:, 2] << 4) | (q_weight_lo[:, 3] << 6)

    q_weight_hi = (q_weight.view(-1, 8) >> 2) & 0x1
    q_weight_hi = (q_weight_hi[:, 0] << 0) | (q_weight_hi[:, 1] << 1) | (q_weight_hi[:, 2] << 2) | (q_weight_hi[:, 3] << 3) | (q_weight_hi[:, 4] << 4) | (q_weight_hi[:, 5] << 5) | (q_weight_hi[:, 6] << 6) | (q_weight_hi[:, 7] << 7)

    q_weight = torch.cat((q_weight_lo, q_weight_hi))
    return q_weight


def dequantize_3bit(q_weight: torch.Tensor, scales: torch.Tensor, group_size:int=128) -> torch.Tensor:
    m = scales.shape[0]
    n = scales.shape[1] * group_size
    assert q_weight.numel() == m * n * 3 // 8

    q_weight_lo = q_weight[:m * n // 4].view(-1, 2)
    q_weight_hi = q_weight[m * n // 4:].view(-1)

    q_weight = torch.stack([
        ((q_weight_lo >> 0) & 0x3)[:, 0] - ((q_weight_hi << 2) & 0x4),
        ((q_weight_lo >> 2) & 0x3)[:, 0] - ((q_weight_hi << 1) & 0x4),
        ((q_weight_lo >> 4) & 0x3)[:, 0] - ((q_weight_hi >> 0) & 0x4),
        ((q_weight_lo >> 6) & 0x3)[:, 0] - ((q_weight_hi >> 1) & 0x4),
        ((q_weight_lo >> 0) & 0x3)[:, 1] - ((q_weight_hi >> 2) & 0x4),
        ((q_weight_lo >> 2) & 0x3)[:, 1] - ((q_weight_hi >> 3) & 0x4),
        ((q_weight_lo >> 4) & 0x3)[:, 1] - ((q_weight_hi >> 4) & 0x4),
        ((q_weight_lo >> 6) & 0x3)[:, 1] - ((q_weight_hi >> 5) & 0x4),
    ], dim=-1).to(torch.int8)

    weight = (q_weight.view(-1, group_size).half() * scales.view(-1, 1)).view(m, n)
    return weight


@torch.no_grad()
def test_gemv():
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    M = 4096 * 3
    N = 4096
    fc_fp16 = nn.Linear(N, M, dtype=torch.half)
    # fc_4bit = bnb.nn.Linear4bit(N, M, compress_statistics=False, quant_type='nf4')

    scales = fc_fp16.weight.data.view(M, N // 128, 128).amax(dim=-1) / (2**(3-1) - 1)
    zeros = torch.full((M, N // 128), fill_value=4)

    fc_3bit = QuantLinear(bits=3, group_size=128, infeatures=N, outfeatures=M, bias=True)
    fc_3bit.pack(fc_fp16, scales=scales, zeros=zeros, g_idx=None)

    fc_fp16.cuda()
    fc_3bit.cuda()
    scales = scales.cuda()

    identity = torch.eye(N, dtype=torch.half, device='cuda')
    fc_3bit.bias.data.zero_()
    fc_fp16.weight.data.copy_(fc_3bit(identity).T)   # dequantize
    fc_3bit.bias.data.copy_(fc_fp16.bias.data)

    x = torch.randn(1, N, dtype=torch.half, device='cuda')

    y_3bit = fc_3bit(x)
    y_fp16 = fc_fp16(x)

    q_weight = quantize_3bit(fc_fp16.weight.data, scales=scales, group_size=128)
    dq_weight = dequantize_3bit(q_weight, scales=scales)
    torch.testing.assert_close(dq_weight, fc_fp16.weight.data)

    def fc_cuda(x):
        return gemv_w3.gemv_i3(x, q_weight, scales, fc_3bit.bias)

    y_cuda = fc_cuda(x)
    # print(y_cuda)
    # first qweight: 2.000000 -3.000000 -3.000000 1.000000 -3.000000 -3.000000 -1.000000 2.000000
    # first weight: 0.010399 -0.015598 -0.015598 0.005199 -0.015598 -0.015598 -0.005199 0.010399
    # first scale: 0.005199
    torch.testing.assert_close(y_3bit, y_fp16, rtol=1e-3, atol=2e-3)
    # torch.testing.assert_close(y_cuda, y_fp16, rtol=1e-3, atol=2e-3)

    elapsed_fp16 = Timer('fc_fp16(x)', globals=locals()).timeit(100).mean
    elapsed_3bit = Timer('fc_3bit(x)', globals=locals()).timeit(100).mean
    elapsed_cuda = Timer('fc_cuda(x)', globals=locals()).timeit(100).mean

    nbytes_fp16 = fc_fp16.weight.nbytes + fc_fp16.bias.nbytes + x.nbytes + y_fp16.nbytes
    nbytes_3bit = fc_3bit.qweight.nbytes + fc_3bit.bias.nbytes + scales.nbytes + x.nbytes + y_3bit.nbytes

    print(f'[fp16] elapsed {elapsed_fp16 * 1e6:.2f} us, memory {nbytes_fp16 / 1e6:.3f} MB, bandwidth {nbytes_fp16 / 1e9 / elapsed_fp16:.3f} GB/s')
    print(f'[3bit] elapsed {elapsed_3bit * 1e6:.2f} us, memory {nbytes_3bit / 1e6:.3f} MB, bandwidth {nbytes_3bit / 1e9 / elapsed_3bit:.3f} GB/s')
    print(f'[cuda] elapsed {elapsed_cuda * 1e6:.2f} us, memory {nbytes_3bit / 1e6:.3f} MB, bandwidth {nbytes_3bit / 1e9 / elapsed_cuda:.3f} GB/s')


if __name__ == '__main__':
    test_gemv()
