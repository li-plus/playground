#include "common.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

int main() {
    constexpr size_t N = 256 * MB;

    float *hA, *hB;
    CHECK_CUDA(cudaHostAlloc(&hA, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hB, N * sizeof(float), cudaHostAllocDefault));

    float *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, N * sizeof(float)));

    auto bench_fn = [=] {
        cudaMemcpyAsync(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(hB, dB, N * sizeof(float), cudaMemcpyDeviceToHost);
    };
    float elapsed_ms = timeit(bench_fn, 2, 10);
    constexpr float PCIE_BW = 64; // uni-directional 64GB/s
    float bw_actual = 2 * N * sizeof(float) / 1e9 / (elapsed_ms / 1e3);
    float bw_util = bw_actual / PCIE_BW;
    printf("pcie bandwidth %.2f / %.2f GB/s (%.2f%%)\n", bw_actual, PCIE_BW, bw_util * 100);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hB));

    return 0;
}
