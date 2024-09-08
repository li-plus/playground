// GEMV: GEneric Matrix-Vector product
// Compute y = Ax, where
// A is a [M, N] matrix, x is a [N] vector, and y is a [M] vector

#include "common.h"

template <int block_size>
__global__ void sgemv_cuda_kernel(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y,
                                  int M, int N) {
    float4 sum4 = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int j = threadIdx.x * 4; j < N; j += blockDim.x * 4) {
        float4 A4 = *(float4 *)&A[blockIdx.x * N + j];
        float4 x4 = *(float4 *)&x[j];
        sum4.x += A4.x * x4.x;
        sum4.y += A4.y * x4.y;
        sum4.z += A4.z * x4.z;
        sum4.w += A4.w * x4.w;
    }

    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    sum = block_reduce_sum<block_size, false>(sum);

    if (threadIdx.x == 0) {
        y[blockIdx.x] = sum;
    }
}

static inline void sgemv_cuda(const float *A, const float *x, float *y, int M, int N) {
    constexpr int num_threads = 512;
    const int num_blocks = M;
    sgemv_cuda_kernel<num_threads><<<num_blocks, num_threads>>>(A, x, y, M, N);
}

static inline void sgemv_cublas(cublasHandle_t handle, const float *A, const float *x, float *y, int M, int N) {
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, A, N, x, 1, &beta, y, 1));
}

int main() {
    constexpr size_t M = 4096;
    constexpr size_t N = 4096 * 4;

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    float *h_A, *h_x, *h_y, *h_y_ref;
    CHECK_CUDA(cudaMallocHost(&h_A, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_x, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y, M * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_ref, M * sizeof(float), cudaHostAllocDefault));

    float *d_A, *d_x, *d_y, *d_y_ref;
    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y_ref, M * sizeof(float)));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i] = uniform(-1, 1);
        }
    }
    for (int i = 0; i < N; i++) {
        h_x[i] = uniform(-1, 1);
    }
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // cuda implementation
    sgemv_cuda(d_A, d_x, d_y, M, N);
    CHECK_CUDA(cudaMemcpy(h_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    // cublas implementation
    sgemv_cublas(cublas_handle, d_A, d_x, d_y_ref, M, N);
    CHECK_CUDA(cudaMemcpy(h_y_ref, d_y_ref, M * sizeof(float), cudaMemcpyDeviceToHost));

    // check results
    check_is_close(h_y, h_y_ref, M, 1e-3, 1e-3);

    // benchmark
    constexpr float nbytes = (M * N + M + N) * sizeof(float);
    const float elapsed_cuda = timeit([=] { sgemv_cuda(d_A, d_x, d_y, M, N); }, 2, 10);
    const float elapsed_cublas = timeit([=] { sgemv_cublas(cublas_handle, d_A, d_x, d_y_ref, M, N); }, 2, 10);

    printf("[sgemv-cuda]   elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_cuda * 1e6f, nbytes / 1e9f / elapsed_cuda);
    printf("[sgemv-cublas] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_cublas * 1e6f,
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
