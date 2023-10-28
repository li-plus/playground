#include "common.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

__global__ void add_1_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    C[i] = A[i] + B[i];
}

__global__ void add_2_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int N) {
    int i = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
    if (i >= N) {
        return;
    }
    float4 a4 = *(float4 *)&A[i];
    float4 b4 = *(float4 *)&B[i];
    float4 c4 = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    *(float4 *)&C[i] = c4;
}

static void add_1(const float *A, const float *B, float *C, int N) {
    constexpr int num_threads = 1024;
    const int num_blocks = N / num_threads;
    add_1_kernel<<<num_blocks, num_threads>>>(A, B, C, N);
}

static void add_2(const float *A, const float *B, float *C, int N) {
    constexpr int num_threads = 1024;
    const int num_blocks = N / num_threads / 4;
    add_2_kernel<<<num_blocks, num_threads>>>(A, B, C, N);
}

static void ref_add(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr size_t N = 128 * MB;

    float *hA, *hB, *hC;
    CHECK_CUDA(cudaHostAlloc(&hA, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hB, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hC, N * sizeof(float), cudaHostAllocDefault));

    for (int i = 0; i < N; i++) {
        hA[i] = i;
        hB[i] = 1;
    }

    float *hC_ref = (float *)malloc(N * sizeof(float));

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<std::pair<std::string, decltype(&add_1)>> kernels{
        {"add_1", &add_1},
        {"add_2", &add_2},
    };

    // launch cuda kernel
    for (const auto &item : kernels) {
        const std::string &name = item.first;
        const auto fn = item.second;

        fn(dA, dB, dC, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost));

        ref_add(hA, hB, hC_ref, N);

        // check results
        for (int i = 0; i < N; i++) {
            if (!is_close(hC[i], hC_ref[i])) {
                printf("value diff: %f vs %f\n", hC[i], hC_ref[i]);
            }
        }

        // perf
        auto bench_fn = [=] { fn(dA, dB, dC, N); };
        float elapsed_ms = timeit(bench_fn, 2, 10);
        float bw_peak = V100SXM2Spec::PEAK_MEM_BW;
        float bw_actual = 3 * N * sizeof(float) / 1e9 / (elapsed_ms / 1000);
        float bw_util = bw_actual / bw_peak;
        printf("[%s] elapsed %.3f ms, bandwidth %.3f GB/s / peak %.3f GB/s (%.3f%%)\n", name.c_str(), elapsed_ms,
               bw_actual, bw_peak, bw_util * 100);
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hB));
    CHECK_CUDA(cudaFreeHost(hC));
    free(hC_ref);

    return 0;
}
