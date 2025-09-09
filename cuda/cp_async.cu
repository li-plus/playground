#include "common.h"
#include <cooperative_groups/memcpy_async.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <functional>
#include <unordered_map>

namespace cg = cooperative_groups;

constexpr int ILP = 128;
constexpr int BLOCK_SIZE = 256;

__global__ void naive_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    __shared__ float s_buffer[1024];
    const int tx = threadIdx.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = tid * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        *(float4 *)&s_buffer[tx * 4] = *(float4 *)&input[i];
        __syncthreads();
        *(float4 *)&output[i] = *(float4 *)&s_buffer[(blockDim.x - 1 - tx) * 4];
        __syncthreads();
    }
}

void naive_cuda(const float *input, float *output, int N) {
    naive_kernel<<<N / BLOCK_SIZE / ILP, BLOCK_SIZE>>>(input, output, N);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void cg_memcpy_async_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    cg::thread_block group = cg::this_thread_block();
    __shared__ float s_buffer[1024];
    for (int i = 0; i < N; i += blockDim.x * gridDim.x * 4) {
        cg::memcpy_async(group, (float4 *)s_buffer, (float4 *)&input[i + bx * blockDim.x * 4], 1024 * sizeof(float));
        cg::wait(group);
        *(float4 *)&output[i + (bx * blockDim.x + tx) * 4] = *(float4 *)&s_buffer[(blockDim.x - 1 - tx) * 4];
        __syncthreads();
    }
}

void cg_memcpy_async_cuda(const float *input, float *output, int N) {
    cg_memcpy_async_kernel<<<N / BLOCK_SIZE / ILP, BLOCK_SIZE>>>(input, output, N);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void cuda_memcpy_async_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int tid = bx * blockDim.x + tx;
    auto pipe = cuda::make_pipeline();

    __shared__ float s_buffer[1024];
    for (int i = tid * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        pipe.producer_acquire();
        cuda::memcpy_async((float4 *)&s_buffer[tx * 4], (float4 *)&input[i], sizeof(float4), pipe);
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();

        *(float4 *)&output[i] = *(float4 *)&s_buffer[(blockDim.x - 1 - tx) * 4];
        __syncthreads();
        pipe.consumer_release();
    }
}

void cuda_memcpy_async_cuda(const float *input, float *output, int N) {
    cuda_memcpy_async_kernel<<<N / BLOCK_SIZE / ILP, BLOCK_SIZE>>>(input, output, N);
    CHECK_CUDA(cudaGetLastError());
}

int main() {
    constexpr size_t N = 128ull * 1024 * 1024;

    float *h_input;
    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_output_ref, *d_output_opt;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_ref, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_opt, N * sizeof(float)));

    // set inputs
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // compute
    naive_cuda(d_input, d_output_ref, N);

    std::unordered_map<std::string, std::function<void(const float *, float *, int)>> ops = {
        {"naive", naive_cuda},
        {"cg_memcpy_async", cg_memcpy_async_cuda},
        {"cuda_memcpy_async", cuda_memcpy_async_cuda},
    };

    std::unordered_map<std::string, float> elapsed_times;

    for (const auto &item : ops) {
        const auto &name = item.first;
        const auto &op = item.second;

        CHECK_CUDA(cudaMemsetAsync(d_output_opt, 0, N * sizeof(float)));
        op(d_input, d_output_opt, N);

        // check results
        check_is_close_d(d_output_ref, d_output_opt, N);

        // benchmark
        const float elapsed = timeit([=] { op(d_input, d_output_opt, N); }, 2, 10);
        elapsed_times.emplace(name, elapsed);
    }

    for (const auto &[name, elapsed] : elapsed_times) {
        const float bandwidth = 2 * N * sizeof(float) / 1e9 / elapsed;
        printf("[%s] elapsed %.3f us, bandwidth %.3f GB/s\n", name.c_str(), elapsed * 1e6, bandwidth);
    }

    CHECK_CUDA(cudaFreeHost(h_input));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_ref));
    CHECK_CUDA(cudaFree(d_output_opt));

    return 0;
}
