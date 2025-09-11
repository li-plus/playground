// GEMV: GEneric Matrix-Vector product
// Compute y = Ax, where
// A is a [M, N] matrix, x is a [N] vector, and y is a [M] vector

#include "common.h"

template <int block_size>
__global__ void hgemv_cuda_kernel(const half *__restrict__ A, const half *__restrict__ x, half *__restrict__ y, int M,
                                  int N) {
    float sum = 0.f;

    for (int j = threadIdx.x * 8; j < N; j += blockDim.x * 8) {
        float4 A_h8 = *(float4 *)&A[blockIdx.x * N + j];
        float4 x_h8 = *(float4 *)&x[j];

#pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 f2 = __half22float2(__hmul2(((half2 *)&A_h8)[i], ((half2 *)&x_h8)[i]));
            sum += f2.x + f2.y;
        }
    }

    sum = block_reduce_sum<block_size, false>(sum);

    if (threadIdx.x == 0) {
        y[blockIdx.x] = __float2half(sum);
    }
}

static inline void hgemv_cuda(const half *A, const half *x, half *y, int M, int N) {
    constexpr int num_threads = 512;
    const int num_blocks = M;
    hgemv_cuda_kernel<num_threads><<<num_blocks, num_threads>>>(A, x, y, M, N);
}

static inline void hgemv_cublas(cublasHandle_t handle, const half *A, const half *x, half *y, int M, int N) {
    const half alpha = __float2half(1);
    const half beta = __float2half(0);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, N, &alpha, A, N, x, N, &beta, y, M));
}

int main() {
    constexpr size_t M = 4096;
    constexpr size_t N = 4096 * 4;

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    half *h_A, *h_x, *h_y, *h_y_ref;
    CHECK_CUDA(cudaMallocHost(&h_A, M * N * sizeof(half), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_x, N * sizeof(half), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y, M * sizeof(half), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_ref, M * sizeof(half), cudaHostAllocDefault));

    half *d_A, *d_x, *d_y, *d_y_ref;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y, M * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_y_ref, M * sizeof(half)));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i] = __float2half(uniform(-0.5, 0.5));
        }
    }
    for (int i = 0; i < N; i++) {
        h_x[i] = __float2half(uniform(-0.5, 0.5));
    }
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, M * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, N * sizeof(half), cudaMemcpyHostToDevice));

    // cuda implementation
    hgemv_cuda(d_A, d_x, d_y, M, N);
    CHECK_CUDA(cudaMemcpy(h_y, d_y, M * sizeof(half), cudaMemcpyDeviceToHost));

    // cublas implementation
    hgemv_cublas(cublas_handle, d_A, d_x, d_y_ref, M, N);
    CHECK_CUDA(cudaMemcpy(h_y_ref, d_y_ref, M * sizeof(half), cudaMemcpyDeviceToHost));

    // check results
    check_is_close(h_y, h_y_ref, M, 1e-2, 1e-3);

    // benchmark
    constexpr float nbytes = (M * N + M + N) * sizeof(half);
    const float elapsed_cuda = timeit([=] { hgemv_cuda(d_A, d_x, d_y, M, N); }, 2, 10);
    const float elapsed_cublas = timeit([=] { hgemv_cublas(cublas_handle, d_A, d_x, d_y_ref, M, N); }, 2, 10);

    printf("[hgemv-cuda]   elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_cuda * 1e6f, nbytes / 1e9f / elapsed_cuda);
    printf("[hgemv-cublas] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_cublas * 1e6f,
           nbytes / 1e9f / elapsed_cublas);

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_y));
    CHECK_CUDA(cudaFreeHost(h_y_ref));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_y_ref));

    return 0;
}
