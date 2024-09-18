#include "common.h"

template <int block_size>
__global__ void any_row_wise_kernel(const bool *input, bool *output, int N) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int num_warps = block_size / WARP_SIZE;

    const int row_idx = blockIdx.x * num_warps + warp_id;
    const bool *input_row = input + row_idx * N;

    bool any = false;
    for (int i = 4 * lane_id; i < N; i += WARP_SIZE * 4) {
        char4 v = *(char4 *)&input_row[i];
        any |= v.x | v.y | v.z | v.w;
    }
    any = __any_sync(FULL_MASK, any);

    if (lane_id == 0) {
        output[row_idx] = any;
    }
}

cudaError_t any_row_wise_cuda(const bool *input, bool *output, int M, int N) {
    constexpr int block_size = 256;
    constexpr int num_warps = block_size / WARP_SIZE;
    any_row_wise_kernel<block_size><<<M / num_warps, block_size>>>(input, output, N);
    return cudaGetLastError();
}

void any_row_wise_cpu(const bool *input, bool *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        bool any = false;
        for (int j = 0; j < N; j++) {
            any |= input[i * N + j];
        }
        output[i] = any;
    }
}

void all_row_wise_cpu(const bool *input, bool *output, int M, int N) {
    for (int i = 0; i < M; i++) {
        bool all = true;
        for (int j = 0; j < N; j++) {
            all &= input[i * N + j];
        }
        output[i] = all;
    }
}

template <int block_size>
__global__ void all_row_wise_kernel(const bool *input, bool *output, int N) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int num_warps = block_size / WARP_SIZE;

    const int row_idx = blockIdx.x * num_warps + warp_id;
    const bool *input_row = input + row_idx * N;

    bool all = true;
    for (int i = 4 * lane_id; i < N; i += WARP_SIZE * 4) {
        char4 v = *(char4 *)&input_row[i];
        all &= v.x & v.y & v.z & v.w;
    }
    all = __all_sync(FULL_MASK, all);

    if (lane_id == 0) {
        output[row_idx] = all;
    }
}

cudaError_t all_row_wise_cuda(const bool *input, bool *output, int M, int N) {
    constexpr int block_size = 256;
    constexpr int num_warps = block_size / WARP_SIZE;
    all_row_wise_kernel<block_size><<<M / num_warps, block_size>>>(input, output, N);
    return cudaGetLastError();
}

int main() {
    const int M = 512;
    const int N = 1024;

    bool *h_input, *h_output_expect, *h_output_actual;
    CHECK_CUDA(cudaMallocHost(&h_input, M * N));
    CHECK_CUDA(cudaMallocHost(&h_output_expect, M));
    CHECK_CUDA(cudaMallocHost(&h_output_actual, M));

    bool *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, M * N));
    CHECK_CUDA(cudaMalloc(&d_output, M));

    // any
    {
        for (int i = 0; i < M * N; i++) {
            h_input[i] = uniform() < 0.001;
        }
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, M * N, cudaMemcpyHostToDevice));

        // cpu
        any_row_wise_cpu(h_input, h_output_expect, M, N);

        // cuda
        CHECK_CUDA(cudaMemsetAsync(d_output, 0xff, M));
        CHECK_CUDA(any_row_wise_cuda(d_input, d_output, M, N));
        CHECK_CUDA(cudaMemcpy(h_output_actual, d_output, M, cudaMemcpyDeviceToHost));

        CHECK(memcmp(h_output_actual, h_output_expect, M) == 0);

        {
            const float elapsed = timeit([&] { any_row_wise_cpu(h_input, h_output_expect, M, N); }, 10, 100);
            printf("[any]  cpu elapsed: %.3f us\n", elapsed * 1e6);
        }
        {
            const float elapsed = timeit([&] { CHECK_CUDA(any_row_wise_cuda(d_input, d_output, M, N)); }, 10, 100);
            printf("[any] cuda elapsed: %.3f us\n", elapsed * 1e6);
        }
    }

    // all
    {
        for (int i = 0; i < M * N; i++) {
            h_input[i] = uniform() < 0.999;
        }
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, M * N, cudaMemcpyHostToDevice));

        // cpu
        all_row_wise_cpu(h_input, h_output_expect, M, N);

        // cuda
        CHECK_CUDA(cudaMemsetAsync(d_output, 0xff, M));
        CHECK_CUDA(all_row_wise_cuda(d_input, d_output, M, N));
        CHECK_CUDA(cudaMemcpy(h_output_actual, d_output, M, cudaMemcpyDeviceToHost));

        CHECK(memcmp(h_output_actual, h_output_expect, M) == 0);

        {
            const float elapsed = timeit([&] { all_row_wise_cpu(h_input, h_output_expect, M, N); }, 10, 100);
            printf("[all]  cpu elapsed: %.3f us\n", elapsed * 1e6);
        }
        {
            const float elapsed = timeit([&] { CHECK_CUDA(all_row_wise_cuda(d_input, d_output, M, N)); }, 10, 100);
            printf("[all] cuda elapsed: %.3f us\n", elapsed * 1e6);
        }
    }

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_expect));
    CHECK_CUDA(cudaFreeHost(h_output_actual));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}