#include "common.h"
#include <cuda_runtime.h>

__global__ void add(float *A, float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

static void launch_add(float *A, float *B, float *C, int N) {
    constexpr int num_threads = 1024;
    const int num_blocks = N / num_threads;
    add<<<num_blocks, num_threads>>>(A, B, C, N);
}

static void ref_add(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr size_t MB = 1024ull * 1024ull;
    constexpr size_t GB = 1024ull * MB;

    constexpr size_t N = 1 * GB;

    float *hA, *hB, *hC;
    CHECK_CUDA(cudaHostAlloc(&hA, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hB, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hC, N * sizeof(float), cudaHostAllocDefault));

    for (int i = 0; i < N; i++) {
        hA[i] = i;
        hB[i] = i - 1;
    }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice));

    // launch cuda kernel
    launch_add(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost));

    // run cpu ref impl
    float *ref_C = (float *)malloc(N * sizeof(float));
    ref_add(hA, hB, ref_C, N);

    // check results
    for (int i = 0; i < N; i++) {
        if (!is_close(hC[i], ref_C[i])) {
            printf("value diff: %f vs %f\n", hC[i], ref_C[i]);
        }
    }

    free(ref_C);

    // perf
    auto fn = [=] { launch_add(dA, dB, dC, N); };
    float elapsed_ms = timeit(fn, 2, 10);
    float bw_peak = 900; // V100 900GB/s
    float bw_actual = 3 * N * sizeof(float) / (float) GB / (elapsed_ms / 1000);
    float bw_util = bw_actual / bw_peak;
    printf("elapsed %.3f ms, bandwidth %.3f GB/s / peak %.3f GB/s (%.3f%%)\n", elapsed_ms, bw_actual, bw_peak,
           bw_util * 100);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hB));
    CHECK_CUDA(cudaFreeHost(hC));
    return 0;
}
