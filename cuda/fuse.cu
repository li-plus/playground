#include "common.h"

// separate mul & add kernels
__global__ void mul_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                float *__restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] * other[idx];
}

__global__ void add_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                float *__restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] + other[idx];
}

void naive_mul_add_cuda(const float *input, const float *alpha, const float *beta, float *output, int n) {
    mul_cuda_kernel<<<n / 128, 128>>>(input, alpha, output);
    add_cuda_kernel<<<n / 128, 128>>>(output, beta, output);
}

// fused mul & add
__global__ void fused_mul_add_cuda_kernel(const float *__restrict__ input, const float *__restrict__ alpha,
                                          const float *__restrict__ beta, float *__restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = std::fma(input[idx], alpha[idx], beta[idx]);
}

void fused_mul_add_cuda(const float *input, const float *alpha, const float *beta, float *output, int n) {
    fused_mul_add_cuda_kernel<<<n / 128, 128>>>(input, alpha, beta, output);
}

int main() {
    const int n = 1024 * 256;

    float *h_input, *h_alpha, *h_beta, *h_output_naive, *h_output_fused;

    CHECK_CUDA(cudaMallocHost(&h_input, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_alpha, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_beta, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_naive, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_fused, n * sizeof(float)));

    float *d_input, *d_alpha, *d_beta, *d_output_naive, *d_output_fused;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alpha, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_fused, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        h_input[i] = uniform();
        h_alpha[i] = uniform();
        h_beta[i] = uniform();
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_alpha, h_alpha, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, h_beta, n * sizeof(float), cudaMemcpyHostToDevice));

    fused_mul_add_cuda(d_input, d_alpha, d_beta, d_output_fused, n);
    CHECK_CUDA(cudaMemcpy(h_output_fused, d_output_fused, n * sizeof(float), cudaMemcpyDeviceToHost));

    naive_mul_add_cuda(d_input, d_alpha, d_beta, d_output_naive, n);
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, n * sizeof(float), cudaMemcpyDeviceToHost));

    // check correctness
    for (int i = 0; i < n; i++) {
        CHECK(is_close(h_output_fused[i], h_output_naive[i]))
            << h_input[i] << " * " << h_alpha[i] << " + " << h_beta[i] << " = " << h_output_fused[i] << " vs "
            << h_output_naive[i];
    }

    const float naive_elapsed =
        timeit([=] { naive_mul_add_cuda(d_input, d_alpha, d_beta, d_output_naive, n); }, 100, 10000);
    const float fused_elapsed =
        timeit([=] { fused_mul_add_cuda(d_input, d_alpha, d_beta, d_output_fused, n); }, 100, 10000);

    printf("naive: %.3f us\n", naive_elapsed * 1e6f);
    printf("fused: %.3f us\n", fused_elapsed * 1e6f);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_output_fused));

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_alpha));
    CHECK_CUDA(cudaFreeHost(h_beta));
    CHECK_CUDA(cudaFreeHost(h_output_naive));
    CHECK_CUDA(cudaFreeHost(h_output_fused));

    return 0;
}