#include "common.h"

constexpr int N = 4096;
constexpr int BLOCK_SIZE = 128;

__global__ void interleaved_cuda_kernel(const float *__restrict__ input, float *__restrict__ output) {
#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        output[s + threadIdx.x] = logf(expf(cosf(sinf(input[s + threadIdx.x]))));
    }
}

void interleaved_cuda(const float *input, float *output) { interleaved_cuda_kernel<<<1, BLOCK_SIZE>>>(input, output); }

__global__ void sequential_cuda_kernel(const float *__restrict__ input, float *__restrict__ output) {
    float reg[N / BLOCK_SIZE];

#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        reg[s / BLOCK_SIZE] = input[s + threadIdx.x];
    }

#pragma unroll
    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        reg[i] = logf(expf(cosf(sinf(reg[i]))));
    }

#pragma unroll
    for (int s = 0; s < N; s += BLOCK_SIZE) {
        output[s + threadIdx.x] = reg[s / BLOCK_SIZE];
    }
}

void sequential_cuda(const float *input, float *output) { sequential_cuda_kernel<<<1, BLOCK_SIZE>>>(input, output); }

int main() {
    float *h_input, *h_output_int, *h_output_seq;

    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_int, N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_seq, N * sizeof(float)));

    float *d_input, *d_output_int, *d_output_seq;

    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_int, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_seq, N * sizeof(float)));

    for (int i = 0; i < N; i++) {
        h_input[i] = uniform();
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    interleaved_cuda(d_input, d_output_int);
    CHECK_CUDA(cudaMemcpy(h_output_int, d_output_int, N * sizeof(float), cudaMemcpyDeviceToHost));

    sequential_cuda(d_input, d_output_seq);
    CHECK_CUDA(cudaMemcpy(h_output_seq, d_output_seq, N * sizeof(float), cudaMemcpyDeviceToHost));

    // check correctness
    check_is_close(h_output_int, h_output_seq, N);

    const float int_elapsed = timeit([=] { interleaved_cuda(d_input, d_output_int); }, 100, 10000);
    const float seq_elapsed = timeit([=] { sequential_cuda(d_input, d_output_seq); }, 100, 10000);

    printf("interleaved: %.3f us\n", int_elapsed * 1e6f);
    printf("sequential:  %.3f us\n", seq_elapsed * 1e6f);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output_int));
    CHECK_CUDA(cudaFreeHost(h_output_seq));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_int));
    CHECK_CUDA(cudaFree(d_output_seq));

    return 0;
}