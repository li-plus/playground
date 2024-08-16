// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

#include "common.h"

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_naive_kernel(float *odata, const float *idata, int M, int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * M + (y + j)] = idata[(y + j) * N + x];
}

void transpose_naive_cuda(float *odata, const float *idata, int M, int N) {
    // idata: [m, n], odata: [n, m]
    const dim3 threads(TILE_DIM, BLOCK_ROWS);
    const dim3 blocks(N / TILE_DIM, M / TILE_DIM);
    transpose_naive_kernel<<<blocks, threads>>>(odata, idata, M, N);
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

void transpose_coalesced_cuda(float *odata, const float *idata, int M, int N) {
    // idata: [m, n], odata: [n, m]
    const dim3 threads(TILE_DIM, BLOCK_ROWS);
    const dim3 blocks(N / TILE_DIM, M / TILE_DIM);
    transpose_coalesced_kernel<<<blocks, threads>>>(odata, idata, M, N);
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

void transpose_swizzle_cuda(float *odata, const float *idata, int M, int N) {
    // idata: [m, n], odata: [n, m]
    const dim3 threads(TILE_DIM, BLOCK_ROWS);
    const dim3 blocks(N / TILE_DIM, M / TILE_DIM);
    transpose_swizzle_kernel<<<blocks, threads>>>(odata, idata, M, N);
}

int main() {
    constexpr size_t M = 1024;
    constexpr size_t N = 1024;

    float *h_input, *h_output_naive, *h_output_coalesced, *h_output_swizzle;
    CHECK_CUDA(cudaHostAlloc(&h_input, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_naive, N * M * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_coalesced, N * M * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_swizzle, N * M * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_output_naive, *d_output_coalesced, *d_output_swizzle;
    CHECK_CUDA(cudaMalloc(&d_input, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, N * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_coalesced, N * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_swizzle, N * M * sizeof(float)));

    // set inputs
    for (size_t i = 0; i < M * N; i++) {
        h_input[i] = uniform();
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // compute
    transpose_naive_cuda(d_output_naive, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    transpose_coalesced_cuda(d_output_coalesced, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_coalesced, d_output_coalesced, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    transpose_swizzle_cuda(d_output_swizzle, d_input, M, N);
    CHECK_CUDA(cudaMemcpy(h_output_swizzle, d_output_swizzle, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // check results
    check_is_close(h_output_coalesced, h_output_naive, N * M);
    check_is_close(h_output_swizzle, h_output_naive, N * M);

    // benchmark
    const float elapsed_naive = timeit([=] { transpose_naive_cuda(d_output_naive, d_input, M, N); }, 2, 10);
    const float bandwidth_naive = 2 * M * N * sizeof(float) / 1e9 / elapsed_naive;
    printf("[naive] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_naive * 1e6, bandwidth_naive);

    const float elapsed_coalesced = timeit([=] { transpose_coalesced_cuda(d_output_coalesced, d_input, M, N); }, 2, 10);
    const float bandwidth_coalesced = 2 * M * N * sizeof(float) / 1e9 / elapsed_coalesced;
    printf("[coalesced] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_coalesced * 1e6, bandwidth_coalesced);

    const float elapsed_swizzle = timeit([=] { transpose_swizzle_cuda(d_output_swizzle, d_input, M, N); }, 2, 10);
    const float bandwidth_swizzle = 2 * M * N * sizeof(float) / 1e9 / elapsed_swizzle;
    printf("[swizzle] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_swizzle * 1e6, bandwidth_swizzle);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_naive));
    CHECK_CUDA(cudaFreeHost(h_output_coalesced));
    CHECK_CUDA(cudaFreeHost(h_output_swizzle));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_output_coalesced));
    CHECK_CUDA(cudaFree(d_output_swizzle));

    return 0;
}
