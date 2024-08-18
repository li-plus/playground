import torch
import torch.nn as nn
# from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
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


def quantize_3bit_v2(weight: torch.Tensor, scales: torch.Tensor, group_size:int=128) -> torch.Tensor:
    """
                t0          t1              t32
    chunk0: [0,  31]    [128, 159]  ... [3968, 3999]
    chunk1: [32, 63]    [160, 191]  ... [4000, 4031]
    chunk2: [64, 95]    [192, 223]  ... [4032, 4063]
    hidden: [96, 127]   [224, 256]  ... [4064, 4096]
    """
    m, n = weight.shape
    assert n % group_size == 0
    q_weight = (weight.view(-1, group_size) / scales.view(-1, 1) + 4).round().clamp(min=0, max=7).to(torch.uint8)
    print(f'quantize_3bit_v2 {q_weight.shape=} {q_weight.sum()=} {q_weight=}')

    q_weight = q_weight.view(m, -1, 4, 32)

    q_weight_packed = q_weight[:, :, :3, 0::2] | (q_weight[:, :, :3, 1::2] << 4)

    q_weight_packed[:, :, 0] |= ((q_weight[:, :, -1, ::2] & 0x1) << (3-0)) | ((q_weight[:, :, -1, 1::2] & 0x1) << (7-0))
    q_weight_packed[:, :, 1] |= ((q_weight[:, :, -1, ::2] & 0x2) << (3-1)) | ((q_weight[:, :, -1, 1::2] & 0x2) << (7-1))
    q_weight_packed[:, :, 2] |= ((q_weight[:, :, -1, ::2] & 0x4) << (3-2)) | ((q_weight[:, :, -1, 1::2] & 0x4) << (7-2))

    return q_weight_packed.transpose(1, 2).contiguous().view(m, -1)


def dequantize_3bit_v2(q_weight_packed: torch.Tensor, scales: torch.Tensor, group_size:int=128) -> torch.Tensor:
    m = q_weight_packed.shape[0]
    n = q_weight_packed.shape[1] * 8 // 3
    assert scales.numel() == m * n // group_size

    q_weight_packed = q_weight_packed.view(m, 3, -1, 16).transpose(1, 2).contiguous()

    q_weight = torch.stack(
        (q_weight_packed & 0x7, (q_weight_packed >> 4) & 0x7), dim=-1
    ).view(m, -1, 3 * 32)

    q_weight_last_chunk = torch.stack(
        (
            ((q_weight_packed[:, :, 0] & 0x08) >> (3-0)) | ((q_weight_packed[:, :, 1] & 0x08) >> (3-1)) | ((q_weight_packed[:, :, 2] & 0x08) >> (3-2)),
            ((q_weight_packed[:, :, 0] & 0x80) >> (7-0)) | ((q_weight_packed[:, :, 1] & 0x80) >> (7-1)) | ((q_weight_packed[:, :, 2] & 0x80) >> (7-2)),
        ), dim=-1
    ).view(m, -1, 32)

    q_weight = torch.cat((q_weight, q_weight_last_chunk), dim=-1).view(-1, group_size)
    print(f'dequantize_3bit_v2 {q_weight.shape=} {q_weight.sum()=} {q_weight=}')

    weight = (q_weight.half() - 4) * scales.view(-1, 1)
    return weight.view(m, n)


def pseudo_quantize_3bit(weight:torch.Tensor, scales: torch.Tensor, group_size:int=128)->torch.Tensor :
    n, k = weight.shape
    q_weight = (weight.view(-1, group_size) / scales.view(-1, 1)).round().clamp(min=-4, max=3)
    dq_weight = q_weight * scales.view(-1, 1)
    return dq_weight.view(n, k)


@torch.no_grad()
def test_gemv():
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    N = 4096 * 3
    K = 4096
    fc_fp16 = nn.Linear(K, N, dtype=torch.half, device='cuda')

    scales = fc_fp16.weight.data.view(N, K // 128, 128).amax(dim=-1) / (2**(3-1) - 1)
    # zeros = torch.full((M, N // 128), fill_value=4)

    fc_fp16.weight.data.copy_(pseudo_quantize_3bit(fc_fp16.weight.data, scales))

    # fc_3bit = QuantLinear(bits=3, group_size=128, infeatures=N, outfeatures=M, bias=True)
    # fc_3bit.pack(fc_fp16, scales=scales, zeros=zeros, g_idx=None)

    # fc_fp16.cuda()
    # fc_3bit.cuda()
    # scales = scales.cuda()

    # identity = torch.eye(N, dtype=torch.half, device='cuda')
    # fc_3bit.bias.data.zero_()
    # fc_fp16.weight.data.copy_(fc_3bit(identity).T)   # dequantize
    # fc_3bit.bias.data.copy_(fc_fp16.bias.data)

    x = torch.randn(1, K, dtype=torch.half, device='cuda')

    print(x)

    q_weight = quantize_3bit_v2(fc_fp16.weight.data, scales=scales)
    dq_weight = dequantize_3bit_v2(q_weight, scales=scales)
    torch.testing.assert_close(dq_weight, fc_fp16.weight.data)

    print(fc_fp16.weight.data[0, :128])

    def fc_cuda(x):
        return gemv_w3.gemv_i3(x, q_weight, scales, fc_fp16.bias)

    # y_3bit = fc_3bit(x)
    y_fp16 = fc_fp16(x)

    y_cuda = fc_cuda(x)

    # print(y_cuda)
    # first qweight: 2.000000 -3.000000 -3.000000 1.000000 -3.000000 -3.000000 -1.000000 2.000000
    # first weight: 0.010399 -0.015598 -0.015598 0.005199 -0.015598 -0.015598 -0.005199 0.010399
    # first scale: 0.005199
    # torch.testing.assert_close(y_3bit, y_fp16, rtol=1e-3, atol=2e-3)
    # torch.testing.assert_close(y_cuda, y_fp16, rtol=1e-3, atol=2e-3)

    elapsed_fp16 = Timer('fc_fp16(x)', globals=locals()).timeit(100).mean
    # elapsed_3bit = Timer('fc_3bit(x)', globals=locals()).timeit(100).mean
    elapsed_cuda = Timer('fc_cuda(x)', globals=locals()).timeit(100).mean

    nbytes_fp16 = fc_fp16.weight.nbytes + fc_fp16.bias.nbytes + x.nbytes + y_fp16.nbytes
    nbytes_3bit = q_weight.nbytes + fc_fp16.bias.nbytes + scales.nbytes + x.nbytes + y_cuda.nbytes

    print(f'[fp16] elapsed {elapsed_fp16 * 1e6:.2f} us, memory {nbytes_fp16 / 1e6:.3f} MB, bandwidth {nbytes_fp16 / 1e9 / elapsed_fp16:.3f} GB/s')
    # print(f'[3bit] elapsed {elapsed_3bit * 1e6:.2f} us, memory {nbytes_3bit / 1e6:.3f} MB, bandwidth {nbytes_3bit / 1e9 / elapsed_3bit:.3f} GB/s')
    print(f'[cuda] elapsed {elapsed_cuda * 1e6:.2f} us, memory {nbytes_3bit / 1e6:.3f} MB, bandwidth {nbytes_3bit / 1e9 / elapsed_cuda:.3f} GB/s')


if __name__ == '__main__':
    test_gemv()
