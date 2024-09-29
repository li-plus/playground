#include "common.h"

__global__ void sleep_kernel() { __nanosleep(1'000'000); }

cudaError_t sleep_cuda(int grid_size, int block_size) {
    sleep_kernel<<<grid_size, block_size>>>();
    return cudaGetLastError();
}

int main() {
    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, 0));
    printf("multiProcessorCount: %d, maxThreadsPerMultiProcessor: %d\n", device_prop.multiProcessorCount,
           device_prop.maxThreadsPerMultiProcessor);

    constexpr int block_size = 128;
    const int concurrent_blocks =
        device_prop.multiProcessorCount * device_prop.maxThreadsPerMultiProcessor / block_size;

    const int grid_sizes[]{concurrent_blocks,         concurrent_blocks + 1, concurrent_blocks * 2,
                           concurrent_blocks * 2 + 1, concurrent_blocks * 3, concurrent_blocks * 3 + 1};

    for (const int grid_size : grid_sizes) {
        const float elapsed = timeit([&] { CHECK_CUDA(sleep_cuda(grid_size, 128)); }, 10, 100);
        printf("[grid=%d] elapsed: %.3f ms\n", grid_size, elapsed * 1e3f);
    }

    return 0;
}
