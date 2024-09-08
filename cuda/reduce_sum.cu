#include "common.h"

template <int block_size>
__global__ void sum_cuda_kernel(const float *__restrict__ input, float *__restrict__ output,
                                float *__restrict__ reduce_buffer, int *__restrict__ semaphore, int N) {
    float4 sum4 = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += 4 * gridDim.x * blockDim.x) {
        float4 v = *(float4 *)&input[i];
        sum4 = make_float4(sum4.x + v.x, sum4.y + v.y, sum4.z + v.z, sum4.w + v.w);
    }
    float sum = (sum4.x + sum4.y) + (sum4.z + sum4.w);
    sum = block_reduce_sum<block_size, false>(sum);

    __shared__ bool is_last_block_done_shared;

    if (threadIdx.x == 0) {
        reduce_buffer[blockIdx.x] = sum;
        __threadfence();
        const int prev_blocks_finished = atomicAdd(semaphore, 1);
        is_last_block_done_shared = (prev_blocks_finished == gridDim.x - 1);
    }
    __syncthreads();

    if (is_last_block_done_shared) {
        sum = 0.f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += reduce_buffer[i];
        }
        sum = block_reduce_sum<block_size, false>(sum);

        if (threadIdx.x == 0) {
            *output = sum;
        }
    }
}

template <int block_size>
static void sum_cuda(const float *input, float *output, float *reduce_buffer, int *semaphore, int N, int num_blocks) {
    CHECK(N % 4 == 0);
    CHECK_CUDA(cudaMemsetAsync(semaphore, 0, sizeof(int)));
    sum_cuda_kernel<block_size><<<num_blocks, block_size>>>(input, output, reduce_buffer, semaphore, N);
}

static float sum_cpu(const float *input, int N) {
    double sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    return sum;
}

int main() {
    constexpr size_t N = 128ull * 1024 * 1024;
    constexpr int num_threads = 1024;
    constexpr int num_blocks = (N + num_threads - 1) / num_threads / 32;

    float *h_input;
    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_output, *d_reduce_buffer;
    int *d_semaphore;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_reduce_buffer, num_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_semaphore, sizeof(int)));

    float h_output_cpu, h_output;

    for (int i = 0; i < N; i++) {
        h_input[i] = uniform(-1, 1);
    }
    h_output_cpu = sum_cpu(h_input, N);

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    sum_cuda<num_threads>(d_input, d_output, d_reduce_buffer, d_semaphore, N, num_blocks);
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // check diff
    CHECK(is_close(h_output, h_output_cpu, 1e-3, 1e-3)) << h_output << " vs " << h_output_cpu;

    // benchmark
    const float elapsed =
        timeit([=] { sum_cuda<num_threads>(d_input, d_output, d_reduce_buffer, d_semaphore, N, num_blocks); }, 2, 10);
    const float bandwidth = N * sizeof(float) / 1e9 / elapsed;
    printf("[reduce_sum] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_reduce_buffer));
    CHECK_CUDA(cudaFree(d_semaphore));

    return 0;
}
