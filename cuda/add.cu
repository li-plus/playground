#include "common.h"

__global__ void add_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                float *__restrict__ output, int N) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        output[i] = input[i] + other[i];
    }
}

static void add_cuda(const float *input, const float *other, float *output, int N) {
    constexpr int num_threads = 1024;
    const int num_blocks = (N + num_threads - 1) / num_threads;
    add_cuda_kernel<<<num_blocks, num_threads>>>(input, other, output, N);
}

static void add_cpu(const float *input, const float *other, float *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = input[i] + other[i];
    }
}

int main() {
    constexpr size_t N = 128ull * 1024 * 1024;

    float *h_input, *h_other, *h_output, *h_output_ref;
    CHECK_CUDA(cudaHostAlloc(&h_input, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_other, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&h_output_ref, N * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_other, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_other, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));

    // set inputs
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
        h_other[i] = 1;
    }

    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_other, h_other, N * sizeof(float), cudaMemcpyHostToDevice));

    // compute
    add_cpu(h_input, h_other, h_output_ref, N);
    add_cuda(d_input, d_other, d_output, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // check results
    for (int i = 0; i < N; i++) {
        CHECK(is_close(h_output[i], h_output_ref[i])) << h_output[i] << " vs " << h_output_ref[i];
    }

    // benchmark
    const float elapsed = timeit([=] { add_cuda(d_input, d_other, d_output, N); }, 2, 10);
    const float bandwidth = 3 * N * sizeof(float) / 1e9 / elapsed;
    printf("[add_cuda] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6, bandwidth);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_other));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFreeHost(h_output_ref));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_other));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
