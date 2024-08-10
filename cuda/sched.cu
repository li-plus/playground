#include "common.h"

constexpr int N = 4096;
constexpr int BLOCK_SIZE = 128;

__global__ void sequential_cuda_kernel(const float *__restrict__ input, float *__restrict__ output) {
#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        output[s + threadIdx.x] = sinf(input[s + threadIdx.x]);
    }
}

void sequential_cuda(const float *input, float *output) { sequential_cuda_kernel<<<1, BLOCK_SIZE>>>(input, output); }

__global__ void parallel_cuda_kernel(const float *__restrict__ input, float *__restrict__ output) {
    float reg[N / BLOCK_SIZE];

#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        reg[s / BLOCK_SIZE] = input[s + threadIdx.x];
    }

#pragma unroll
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        reg[i] = sinf(reg[i]);
    }

#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        output[s + threadIdx.x] = reg[s / BLOCK_SIZE];
    }
}

void parallel_cuda(const float *input, float *output) { parallel_cuda_kernel<<<1, BLOCK_SIZE>>>(input, output); }

int main() {
    float *h_input, *h_output_seq, *h_output_par;

    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_seq, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_par, N * sizeof(float)));

    float *d_input, *d_output_seq, *d_output_par;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_seq, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_par, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_input[i] = uniform();
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    sequential_cuda(d_input, d_output_seq);
    CHECK_CUDA(cudaMemcpy(h_output_seq, d_output_seq, N * sizeof(float), cudaMemcpyDeviceToHost));

    parallel_cuda(d_input, d_output_par);
    CHECK_CUDA(cudaMemcpy(h_output_par, d_output_par, N * sizeof(float), cudaMemcpyDeviceToHost));

    // check correctness
    check_is_close(h_output_seq, h_output_par, N);

    const float seq_elapsed = timeit([=] { sequential_cuda(d_input, d_output_seq); }, 100, 10000);
    const float par_elapsed = timeit([=] { parallel_cuda(d_input, d_output_par); }, 100, 10000);

    printf("sequential: %.3f us\n", seq_elapsed * 1e6f);
    printf("parallel:   %.3f us\n", par_elapsed * 1e6f);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_seq));
    CHECK_CUDA(cudaFreeHost(h_output_par));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_seq));
    CHECK_CUDA(cudaFree(d_output_par));

    return 0;
}