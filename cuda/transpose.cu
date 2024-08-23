// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive_kernel(float *odata, const float *idata, int M, int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * M + (y + j)] = idata[(y + j) * N + x];
}

__global__ void transpose_coalesced_kernel(float *odata, const float *idata, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // prevent bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * N + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void transpose_swizzle_kernel(float *odata, const float *idata, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][(threadIdx.y + j) ^ threadIdx.x] = idata[(y + j) * N + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * M + x] = tile[threadIdx.x][threadIdx.x ^ (threadIdx.y + j)];
}

__global__ void transpose_async_barrier_swizzle_kernel(float *odata, const float *idata, int M, int N) {
    namespace cg = cooperative_groups;

    __shared__ float tile[TILE_DIM][TILE_DIM];

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    auto block = cg::this_thread_block();
    if (block.thread_rank() == 0) {
        init(&barrier, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        cuda::memcpy_async(&tile[threadIdx.y + j][(threadIdx.y + j) ^ threadIdx.x], &idata[(y + j) * N + x],
                           sizeof(float), barrier);

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    barrier.arrive_and_wait();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * M + x] = tile[threadIdx.x][threadIdx.x ^ (threadIdx.y + j)];
}

__global__ void transpose_async_pipeline_swizzle_kernel(float *odata, const float *idata, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        cuda::memcpy_async(&tile[threadIdx.y + j][(threadIdx.y + j) ^ threadIdx.x], &idata[(y + j) * N + x],
                           sizeof(float), pipe);

    pipe.producer_commit();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    pipe.consumer_wait();
    __syncthreads();

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * M + x] = tile[threadIdx.x][threadIdx.x ^ (threadIdx.y + j)];
}

#define make_launcher(launcher, kernel)                                                                                \
    void launcher(float *odata, const float *idata, int M, int N) {                                                    \
        const dim3 threads(TILE_DIM, BLOCK_ROWS);                                                                      \
        const dim3 blocks(N / TILE_DIM, M / TILE_DIM);                                                                 \
        kernel<<<blocks, threads>>>(odata, idata, M, N);                                                               \
    }

make_launcher(transpose_naive_cuda, transpose_naive_kernel);
make_launcher(transpose_coalesced_cuda, transpose_coalesced_kernel);
make_launcher(transpose_swizzle_cuda, transpose_swizzle_kernel);
make_launcher(transpose_async_barrier_swizzle_cuda, transpose_async_barrier_swizzle_kernel);
make_launcher(transpose_async_pipeline_swizzle_cuda, transpose_async_pipeline_swizzle_kernel);

#undef make_launcher

int main() {
    constexpr size_t M = 4096;
    constexpr size_t N = 4096;

    float *h_input, *h_output_ref, *h_output_out;
    CHECK_CUDA(cudaHostAlloc(&h_input, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_ref, N * M * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_out, N * M * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * M * sizeof(float)));

    // set inputs
    for (size_t i = 0; i < M * N; i++) {
        h_input[i] = uniform();
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // compute
    CHECK_CUDA(cudaMemset(d_output, 0, N * M * sizeof(float)));
    transpose_naive_cuda(d_output, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_ref, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMemset(d_output, 0, N * M * sizeof(float)));
    transpose_coalesced_cuda(d_output, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_out, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_output_out, h_output_ref, N * M);

    CHECK_CUDA(cudaMemset(d_output, 0, N * M * sizeof(float)));
    transpose_swizzle_cuda(d_output, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_out, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_output_out, h_output_ref, N * M);

    CHECK_CUDA(cudaMemset(d_output, 0, N * M * sizeof(float)));
    transpose_async_barrier_swizzle_cuda(d_output, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_out, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_output_out, h_output_ref, N * M);

    CHECK_CUDA(cudaMemset(d_output, 0, N * M * sizeof(float)));
    transpose_async_pipeline_swizzle_cuda(d_output, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_out, d_output, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_output_out, h_output_ref, N * M);

    // benchmark
    {
        const float elapsed = timeit([=] { transpose_naive_cuda(d_output, d_input, M, N); }, 2, 10);
        const float bandwidth = 2 * M * N * sizeof(float) / 1e9 / elapsed;
        printf("[naive] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);
    }
    {
        const float elapsed = timeit([=] { transpose_coalesced_cuda(d_output, d_input, M, N); }, 2, 10);
        const float bandwidth = 2 * M * N * sizeof(float) / 1e9 / elapsed;
        printf("[coalesced] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);
    }
    {
        const float elapsed = timeit([=] { transpose_swizzle_cuda(d_output, d_input, M, N); }, 2, 10);
        const float bandwidth = 2 * M * N * sizeof(float) / 1e9 / elapsed;
        printf("[swizzle] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);
    }
    {
        const float elapsed = timeit([=] { transpose_async_barrier_swizzle_cuda(d_output, d_input, M, N); }, 2, 10);
        const float bandwidth = 2 * M * N * sizeof(float) / 1e9 / elapsed;
        printf("[async-barrier] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);
    }
    {
        const float elapsed = timeit([=] { transpose_async_pipeline_swizzle_cuda(d_output, d_input, M, N); }, 2, 10);
        const float bandwidth = 2 * M * N * sizeof(float) / 1e9 / elapsed;
        printf("[async-pipeline] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);
    }

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_ref));
    CHECK_CUDA(cudaFreeHost(h_output_out));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
