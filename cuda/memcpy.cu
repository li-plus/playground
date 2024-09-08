#include "common.h"

__global__ void memcpy_cuda_kernel(void *__restrict__ dst, const void *__restrict__ src, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count / sizeof(float4); i += gridDim.x * blockDim.x) {
        ((float4 *)dst)[i] = ((float4 *)src)[i];
    }
}

void memcpy_cuda(void *dst, const void *src, size_t count) {
    CHECK(count % sizeof(float4) == 0);
    constexpr int num_threads = 1024;
    const int num_blocks = (count + num_threads - 1) / (16 * num_threads);
    memcpy_cuda_kernel<<<num_blocks, num_threads>>>(dst, src, count);
}

int main() {
    constexpr size_t N = 1024ull * 1024 * 1024;

    char *h_a, *h_b;
    CHECK_CUDA(cudaMallocHost(&h_a, N, cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_b, N, cudaHostAllocDefault));

    char *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, N));
    CHECK_CUDA(cudaMalloc(&d_b, N));

    memset(h_b, 0x9c, N);

    const float h2h_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(h_a, h_b, N, cudaMemcpyHostToHost)); }, 2, 10);

    const float h2d_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, N, cudaMemcpyHostToDevice)); }, 2, 10);

    const float d2d_elapsed =
        timeit([=] { CHECK_CUDA(cudaMemcpyAsync(d_b, d_a, N, cudaMemcpyDeviceToDevice)); }, 2, 10);

    CHECK_CUDA(cudaMemset(d_b, 0x00, N));
    const float d2d_cuda_elapsed = timeit([=] { memcpy_cuda(d_b, d_a, N); }, 2, 10);

    const float d2h_elapsed = timeit([=] { CHECK_CUDA(cudaMemcpyAsync(h_b, d_b, N, cudaMemcpyDeviceToHost)); }, 2, 10);

    for (int i = 0; i < N / sizeof(int); i++) {
        CHECK(((int *)h_b)[i] == 0x9c9c9c9c);
    }

    printf("h2h: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, h2h_elapsed, 2 * N / 1e9 / h2h_elapsed);
    printf("h2d: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, h2d_elapsed, 2 * N / 1e9 / h2d_elapsed);
    printf("d2d: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, d2d_elapsed, 2 * N / 1e9 / d2d_elapsed);
    printf("d2d-cuda: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, d2d_cuda_elapsed,
           2 * N / 1e9 / d2d_cuda_elapsed);
    printf("d2h: size %.3f GB, cost %.3f s, bandwidth %.3f GB/s\n", N / 1e9, d2h_elapsed, 2 * N / 1e9 / d2h_elapsed);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));

    return 0;
}
