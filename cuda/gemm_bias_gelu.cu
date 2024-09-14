/*
cublasLt example: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
PyTorch code: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp
*/

#include "common.h"
#include <cublasLt.h>

void gemm_bias_gelu_cublaslt(cublasLtHandle_t handle, const float *input, const float *weight, const float *bias,
                             float *output, int M, int N, int K, void *workspace, size_t workspace_size) {
    const float *A = weight; // [K, N] in row major -> [N, K] in col major
    const float *B = input;  // [M, K] in row major -> [K, M] in col major
    float *C = output;       // [M, N] in row major -> [N, M] in col major = OP_N(A) @ OP_N(B)

    const float alpha = 1.f;
    const float beta = 0.f;

    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans_A = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_A, sizeof(trans_A)));
    cublasOperation_t trans_B = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_B, sizeof(trans_B)));

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    CHECK_CUBLAS(
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    cublasLtMatrixLayout_t A_desc, B_desc, C_desc;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32F, N, M, N));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic_result{};
    int returned_result = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_desc, C_desc, C_desc, preference, 1,
                                                &heuristic_result, &returned_result));
    if (returned_result == 0) {
        CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    CHECK_CUBLAS(cublasLtMatmul(handle, matmul_desc, &alpha, A, A_desc, B, B_desc, &beta, C, C_desc, output, C_desc,
                                &heuristic_result.algo, workspace, workspace_size, 0));

    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(C_desc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmul_desc));
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_inplace_kernel(float *input, const float *bias, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    input[idx] = gelu(input[idx] + bias[idx % N]);
}

cudaError_t bias_gelu_inplace_cuda(float *input, const float *bias, int M, int N) {
    constexpr int block_size = 128;
    const int grid_size = M * N / block_size;
    bias_gelu_inplace_kernel<<<grid_size, block_size>>>(input, bias, N);
    return cudaGetLastError();
}

void gemm_bias_gelu_cublas(cublasHandle_t handle, const float *input, const float *weight, const float *bias,
                           float *output, int M, int N, int K) {
    const float *A = weight;
    const float *B = input;
    float *C = output;
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, A, N, B, K, &beta, C, N));
    CHECK_CUDA(bias_gelu_inplace_cuda(output, bias, M, N));
}

void benchmark(int M, int N, int K) {
    float *h_input, *h_weight, *h_bias, *h_output_expect, *h_output_actual;
    CHECK_CUDA(cudaMallocHost(&h_input, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_weight, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_bias, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_expect, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_actual, M * N * sizeof(float)));

    float *d_input, *d_weight, *d_bias, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weight, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, M * N * sizeof(float)));

    void *workspace;
    const size_t workspace_size = 1024 * 1024;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    for (int i = 0; i < M * K; i++) {
        h_input[i] = uniform(-0.5, 0.5);
    }
    for (int i = 0; i < K * N; i++) {
        h_weight[i] = uniform(-0.5, 0.5);
    }
    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_weight, h_weight, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    cublasLtHandle_t cublaslt_handle;
    CHECK_CUBLAS(cublasLtCreate(&cublaslt_handle));

    {
        CHECK_CUDA(cudaMemsetAsync(d_output, 0, M * N * sizeof(float)));
        gemm_bias_gelu_cublas(cublas_handle, d_input, d_weight, d_bias, d_output, M, N, K);
        CHECK_CUDA(cudaMemcpy(h_output_actual, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    {
        CHECK_CUDA(cudaMemsetAsync(d_output, 0, M * N * sizeof(float)));
        gemm_bias_gelu_cublaslt(cublaslt_handle, d_input, d_weight, d_bias, d_output, M, N, K, workspace,
                                workspace_size);
        CHECK_CUDA(cudaMemcpy(h_output_expect, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    check_is_close(h_output_actual, h_output_expect, M * N, 1e-4, 1e-4);

    const float tflops = (2.f * M * N * K) / 1e12;
    const float gbytes = (M * K + K * N + M * N) / 1e9;
    {
        const float elapsed = timeit(
            [&] { gemm_bias_gelu_cublas(cublas_handle, d_input, d_weight, d_bias, d_output, M, N, K); }, 10, 100);
        printf("[cublas] elapsed: %.3f us, throughput: %.3f TFLOPS, bandwidth: %.3f GB/s\n", elapsed * 1e6f,
               tflops / elapsed, gbytes / elapsed);
    }
    {
        const float elapsed = timeit(
            [&] {
                gemm_bias_gelu_cublaslt(cublaslt_handle, d_input, d_weight, d_bias, d_output, M, N, K, workspace,
                                        workspace_size);
            },
            10, 100);
        printf("[cublasLt] elapsed: %.3f us, throughput: %.3f TFLOPS, bandwidth: %.3f GB/s\n", elapsed * 1e6f,
               tflops / elapsed, gbytes / elapsed);
    }

    CHECK_CUDA(cudaFree(workspace));

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_weight));
    CHECK_CUDA(cudaFreeHost(h_bias));
    CHECK_CUDA(cudaFreeHost(h_output_expect));
    CHECK_CUDA(cudaFreeHost(h_output_actual));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output));
}

int main(int argc, char **argv) {
    int M = 4096, N = 1024, K = 2048;
    benchmark(M, N, K);
    return 0;
}
