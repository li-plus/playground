#include "common.h"

int main() {
    constexpr size_t N = 1024ull * 1024 * 1024;

    char *h_a, *h_b;
    CHECK_CUDA(cudaHostAlloc(&h_a, N, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_b, N, cudaHostAllocDefault));
    memset(h_a, 0x9c, N);
    memset(h_b, 0xc9, N);

    char *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, N));
    CHECK_CUDA(cudaMalloc(&d_b, N));
    CHECK_CUDA(cudaMemset(d_a, 0x9c, N));
    CHECK_CUDA(cudaMemset(d_b, 0xc9, N));

    const float h2h_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(h_a, h_b, N, cudaMemcpyHostToHost)); }, 2, 10);

    const float h2d_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, N, cudaMemcpyHostToDevice)); }, 2, 10);

    const float d2d_elapsed =
        timeit([=] { CHECK_CUDA(cudaMemcpyAsync(d_b, d_a, N, cudaMemcpyDeviceToDevice)); }, 2, 10);

    const float d2h_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(h_b, d_b, N, cudaMemcpyDeviceToHost)); }, 2, 10);

    printf("h2h: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, h2h_elapsed, 2 * N / 1e9 / h2h_elapsed);
    printf("h2d: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, h2d_elapsed, 2 * N / 1e9 / h2d_elapsed);
    printf("d2d: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, d2d_elapsed, 2 * N / 1e9 / d2d_elapsed);
    printf("d2h: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, d2h_elapsed, 2 * N / 1e9 / d2h_elapsed);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));

    return 0;
}
