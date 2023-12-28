import torch
from torch.utils.benchmark import Timer


class A100Spec:
    PEAK_MEM_BW = 1935  # GB/s
    PEAK_TFLOPS = {
        torch.float32: 19.5,
        torch.float16: 312,
    }


class V100Spec:
    PEAK_MEM_BW = 900  # GB/s
    PEAK_TFLOPS = {
        torch.float32: 15.7,
        torch.float16: 125,
    }


def get_device_spec():
    gpu_name = torch.cuda.get_device_name()
    if "V100" in gpu_name:
        return V100Spec()
    elif "A100" in gpu_name:
        return A100Spec()
    else:
        assert False, f"unknown device {gpu_name}"


GPUSpec = get_device_spec()
PEAK_PCIE_BW = 64


def perf_gemm(M, N, K, dtype):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")

    timer = Timer(
        stmt="a @ b",
        globals={"a": a, "b": b},
        label="gemm",
    )

    result = timer.timeit(10)
    tflops_achieved = 2 * M * N * K / 1e12 / result.mean
    tflops_promised = GPUSpec.PEAK_TFLOPS[dtype]
    mfu = tflops_achieved / tflops_promised
    print(f"{dtype} gemm tflops {tflops_achieved:.2f} / {tflops_promised:.2f}, mfu {mfu:.2f}")


def perf_pcie():
    a = torch.zeros(1024 * 1024 * 256, device="cuda").cpu().pin_memory()
    timer = Timer(stmt="a.cuda()", globals={"a": a})
    result = timer.timeit(10)
    bw_achieved = a.numel() * 4 / 1e9 / result.mean
    bw_promised = PEAK_PCIE_BW
    bw_util = bw_achieved / bw_promised
    print(f"pcie bandwidth (GB/s) {bw_achieved:.2f} / {bw_promised:.2f}, util {bw_util:.2f}")


def perf_hbm():
    a = torch.randn(1024 * 1024 * 256, dtype=torch.float32, device="cuda")
    b = torch.randn(1024 * 1024 * 256, dtype=torch.float32, device="cuda")
    timer = Timer(stmt="a.copy_(b)", globals={"a": a, "b": b})
    result = timer.timeit(100)
    bw_achieved = (a.numel() + b.numel()) * 4 / 1e9 / result.mean
    bw_promised = GPUSpec.PEAK_MEM_BW
    bw_util = bw_achieved / bw_promised
    print(f"global memory bandwidth (GB/s) {bw_achieved:.2f} / {bw_promised:.2f}, util {bw_util:.2f}")


@torch.inference_mode()
def main():
    M = N = K = 4096

    perf_gemm(M, N, K, torch.float32)
    perf_gemm(M, N, K, torch.float16)

    perf_pcie()
    perf_hbm()


if __name__ == "__main__":
    main()
