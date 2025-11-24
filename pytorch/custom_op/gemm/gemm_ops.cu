#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void gemm_bf16(cublasHandle_t handle, const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, A,
                                      CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_fp16(cublasHandle_t handle, const half *A, const half *B, half *C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F, N, A,
                                      CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_contiguous() && A.is_cuda() && A.ndimension() == 2);
    TORCH_CHECK(B.is_contiguous() && B.is_cuda() && B.ndimension() == 2);
    TORCH_CHECK(A.size(1) == B.size(0) && A.dtype() == B.dtype());

    const auto M = A.size(0);
    const auto N = B.size(1);
    const auto K = A.size(1);

    auto C = torch::empty({M, N}, A.options());

    auto handle = at::cuda::getCurrentCUDABlasHandle();

    AT_DISPATCH_SWITCH(A.scalar_type(), "gemm", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                           gemm_fp16(handle, (const half *)A.const_data_ptr(), (const half *)B.const_data_ptr(),
                                     (half *)C.mutable_data_ptr(), M, N, K);
                       }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
                           gemm_bf16(handle, (const nv_bfloat16 *)A.const_data_ptr(),
                                     (const nv_bfloat16 *)B.const_data_ptr(), (nv_bfloat16 *)C.mutable_data_ptr(), M, N,
                                     K);
                       }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }
