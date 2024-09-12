#include "common.h"

template <int block_size>
__global__ void sum_block_reduce_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    const float *input_row = input + blockIdx.x * N;

    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += input_row[i];
    }
    sum = block_reduce_sum<block_size, false>(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

cudaError_t sum_block_reduce_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 128;
    const int grid_size = M;
    sum_block_reduce_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    return cudaGetLastError();
}

template <int block_size>
__global__ void sum_warp_reduce_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int row_id = blockIdx.x * (block_size / WARP_SIZE) + warp_id;
    const float *input_row = input + row_id * N;

    float sum = 0.f;
    for (int i = lane_id; i < N; i += WARP_SIZE) {
        sum += input_row[i];
    }
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[row_id] = sum;
    }
}

cudaError_t sum_warp_reduce_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 128;
    const int grid_size = M / (block_size / WARP_SIZE);
    sum_warp_reduce_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    return cudaGetLastError();
}

template <int block_size>
__global__ void sum_cg_warp_reduce_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<WARP_SIZE>(block);

    const int row_id = block.group_index().x * tile.meta_group_size() + tile.meta_group_rank();
    const float *input_row = input + row_id * N;

    float sum = 0.f;
    // replacing WARP_SIZE with tile.num_threads() will disable nvcc loop unrolling, causing bad performance
    for (int i = tile.thread_rank(); i < N; i += WARP_SIZE) {
        sum += input_row[i];
    }
    sum = cg::reduce(tile, sum, cg::plus<float>());

    if (tile.thread_rank() == 0) {
        output[row_id] = sum;
    }
}

cudaError_t sum_cg_warp_reduce_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 128;
    const int grid_size = M / (block_size / WARP_SIZE);
    sum_cg_warp_reduce_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    return cudaGetLastError();
}

int main() {
    constexpr size_t M = 512;
    constexpr size_t N = 2048;

    float *h_input, *h_output_expect, *h_output_actual;
    CHECK_CUDA(cudaMallocHost(&h_input, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_output_expect, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_output_actual, M * N * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, M * N * sizeof(float)));

    for (int i = 0; i < M * N; i++) {
        h_input[i] = uniform(-1, 1);
    }
    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // run & check
    {
        CHECK_CUDA(cudaMemsetAsync(d_output, 0, M * N * sizeof(float)));
        CHECK_CUDA(sum_block_reduce_cuda(d_input, d_output, M, N));
        CHECK_CUDA(cudaMemcpy(h_output_expect, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    auto run_and_check = [&](decltype(sum_block_reduce_cuda) fn) {
        CHECK_CUDA(cudaMemsetAsync(d_output, 0, M * N * sizeof(float)));
        CHECK_CUDA(fn(d_input, d_output, M, N));
        CHECK_CUDA(cudaMemcpy(h_output_actual, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        check_is_close(h_output_expect, h_output_actual, M * N, 1e-3, 1e-3);
    };
    run_and_check(sum_warp_reduce_cuda);
    run_and_check(sum_cg_warp_reduce_cuda);

    // benchmark
    auto benchmark = [&](decltype(sum_block_reduce_cuda) fn, const char *name) {
        constexpr float nbytes = 2 * M * N * sizeof(float);
        const float elapsed = timeit([&] { CHECK_CUDA(fn(d_input, d_output, M, N)); }, 10, 100);
        printf("[%s] elapsed %.3f us, bandwidth %.3f GB/s\n", name, elapsed * 1e6f, nbytes / 1e9f / elapsed);
    };
    benchmark(sum_block_reduce_cuda, "block-reduce");
    benchmark(sum_warp_reduce_cuda, "warp-reduce");
    benchmark(sum_cg_warp_reduce_cuda, "cg-warp-reduce");

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_expect));
    CHECK_CUDA(cudaFreeHost(h_output_actual));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
