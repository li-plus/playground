import torch.utils.cpp_extension
from torch.utils.benchmark import Timer
import pytest

debug = False
extra_cflags = ["-Og", "-g"] if debug else ["-O3"]
extra_cuda_cflags = ["-O0", "-lineinfo"] if debug else ["-O3"]

gemm_ops = torch.utils.cpp_extension.load(
    name="gemm_ops",
    sources=["gemm_ops.cu", "gemm_ops.cpp"],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    build_directory="build/",
    verbose=True,
)

@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_gemm(dtype):
    M = 4096
    N = 4096
    K = 4096

    A_ref = torch.randn(M, K, dtype=dtype, device="cuda")
    B_ref = torch.randn(K, N, dtype=dtype, device="cuda")
    A_opt = A_ref.clone()
    B_opt = B_ref.clone()

    C_ref = A_ref @ B_ref
    C_opt = gemm_ops.gemm(A_opt, B_opt)
    torch.testing.assert_close(C_opt, C_ref)

    fwd_ref = Timer("A_ref @ B_ref", globals={**globals(), **locals()}).timeit(10).mean * 1e6
    fwd_opt = Timer("gemm_ops.gemm(A_opt, B_opt)", globals={**globals(), **locals()}).timeit(10).mean * 1e6
    print(f"ref {fwd_ref:.2f} us, opt {fwd_opt:.2f} us")
