// GEMV: GEneric Matrix-Vector product
// Compute y = Ax, where
// A is a [M, N] matrix, x is a [N] vector, and y is a [M] vector

#include "common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float sum) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask, 32);
    }
    return sum;
}

// assert #threads = 1024
__device__ __forceinline__ float block_reduce_sum(float sum) {
    sum = warp_reduce_sum(sum);

    __shared__ float sums[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    if (lane_id == 0) {
        sums[warp_id] = sum;
    }
    __syncthreads();

    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = sums[lane_id];
        sum = warp_reduce_sum(sum);
    }

    return sum;
}

__global__ void sgemv_kernel(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M,
                             int N) {
    const int i = blockIdx.x;

    float sum = 0.f;

    for (int j = threadIdx.x; j < N; j += WARP_SIZE) {
        sum += A[i * N + j] * x[j];
    }

    sum = warp_reduce_sum(sum);

    if (threadIdx.x == 0) {
        y[i] = sum;
    }
}

static void sgemv(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M, int N) {
    constexpr int num_threads = 32;
    const int num_blocks = M;
    sgemv_kernel<<<num_blocks, num_threads>>>(A, x, y, M, N);
}

__global__ void sgemv_2_kernel(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M,
                               int N) {
    const int i = blockIdx.x;

    float sum = 0.f;

    for (int j = threadIdx.x * 4; j < N; j += WARP_SIZE * 4) {
        float4 A4 = *(float4 *)&A[i * N + j];
        float4 x4 = *(float4 *)&x[j];
        sum += (A4.x * x4.x + A4.y * x4.y) + (A4.z * x4.z + A4.w * x4.w);
    }

    sum = warp_reduce_sum(sum);

    if (threadIdx.x == 0) {
        y[i] = sum;
    }
}

static void sgemv_2(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M, int N) {
    constexpr int num_threads = 32;
    const int num_blocks = M;
    sgemv_2_kernel<<<num_blocks, num_threads>>>(A, x, y, M, N);
}

__global__ void sgemv_3_kernel(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M,
                               int N) {
    const int i = blockIdx.x;

    float sum = 0.f;

    for (int j = threadIdx.x * 4; j < N; j += blockDim.x * 4) {
        float4 A4 = *(float4 *)&A[i * N + j];
        float4 x4 = *(float4 *)&x[j];
        sum += (A4.x * x4.x + A4.y * x4.y) + (A4.z * x4.z + A4.w * x4.w);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        y[i] = sum;
    }
}

static void sgemv_3(const float *__restrict__ A, const float *__restrict__ x, float *__restrict__ y, int M, int N) {
    constexpr int num_threads = 1024;
    const int num_blocks = M;
    sgemv_3_kernel<<<num_blocks, num_threads>>>(A, x, y, M, N);
}

static void ref_sgemv(cublasHandle_t handle, const float *__restrict__ A, const float *__restrict__ x,
                      float *__restrict__ y, int M, int N) {
    float alpha = 1;
    float beta = 0;
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, A, N, x, 1, &beta, y, 1));
}

struct KernelEntry {
    std::string name;
    std::function<void(const float *, const float *, float *, int, int)> fn;
};

int main() {
    constexpr size_t M = 4096;
    constexpr size_t N = 4096 * 4;

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    float *hA, *hx, *out_hy, *ref_hy;
    CHECK_CUDA(cudaHostAlloc(&hA, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hx, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&out_hy, M * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&ref_hy, M * sizeof(float), cudaHostAllocDefault));

    float *dA, *dx, *out_dy, *ref_dy;
    CHECK_CUDA(cudaMalloc(&dA, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dx, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_dy, M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ref_dy, M * sizeof(float)));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            hA[i] = uniform();
        }
    }
    for (int i = 0; i < N; i++) {
        hx[i] = uniform();
    }
    CHECK_CUDA(cudaMemcpy(dA, hA, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));

    KernelEntry kernels[]{
        {"sgemv", sgemv},
        {"sgemv_2", sgemv_2},
        {"sgemv_3", sgemv_3},
    };

    for (const auto &kernel : kernels) {
        // cuda implementation
        kernel.fn(dA, dx, out_dy, M, N);
        CHECK_CUDA(cudaMemcpy(out_hy, out_dy, M * sizeof(float), cudaMemcpyDeviceToHost));

        // cublas reference implementation
        ref_sgemv(cublas_handle, dA, dx, ref_dy, M, N);
        CHECK_CUDA(cudaMemcpy(ref_hy, ref_dy, M * sizeof(float), cudaMemcpyDeviceToHost));

        // check results
        for (int i = 0; i < M; i++) {
            if (!is_close(out_hy[i], ref_hy[i])) {
                printf("[%s] value diff: %f vs %f\n", kernel.name.c_str(), out_hy[i], ref_hy[i]);
            }
        }
    }

    // perf
    KernelEntry benchmark_kernels[]{{"sgemv", sgemv},
                                    {"sgemv_2", sgemv_2},
                                    {"sgemv_3", sgemv_3},
                                    {"cublas", [=](const float *A, const float *x, float *y, int M, int N) {
                                         ref_sgemv(cublas_handle, A, x, y, M, N);
                                     }}};

    constexpr float bw_peak = 900; // V100 900GB/s
    constexpr float total_mem = (M * N + M + N) * sizeof(float) / (float)GB;
    for (const auto &kernel : benchmark_kernels) {
        auto fn = [=] { kernel.fn(dA, dx, out_dy, M, N); };
        const float elapsed_ms = timeit(fn, 2, 10);
        const float bw_actual = total_mem / (elapsed_ms / 1000);
        const float bw_util = bw_actual / bw_peak;
        printf("[%s] elapsed %.3f ms, bandwidth %.3f GB/s out of peak %.3f GB/s (%.3f%%)\n", kernel.name.c_str(),
               elapsed_ms, bw_actual, bw_peak, bw_util * 100);
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(out_dy));
    CHECK_CUDA(cudaFree(ref_dy));

    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hx));
    CHECK_CUDA(cudaFreeHost(out_hy));
    CHECK_CUDA(cudaFreeHost(ref_hy));

    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
