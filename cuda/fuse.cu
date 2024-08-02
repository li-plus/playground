#include "common.h"

// separate mul & add kernels
__global__ void mul_cuda_kernel(const float *input, const float *other, float *output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] * other[idx];
}

__global__ void add_cuda_kernel(const float *input, const float *other, float *output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = input[idx] + other[idx];
}

void naive_mul_add_cuda(const float *input, const float *alpha, const float *beta, float *output, int n) {
    mul_cuda_kernel<<<n / 128, 128>>>(input, alpha, output);
    add_cuda_kernel<<<n / 128, 128>>>(output, beta, output);
}

// fused mul & add
__global__ void fused_mul_add_cuda_kernel(const float *input, const float *alpha, const float *beta, float *output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = std::fma(input[idx], alpha[idx], beta[idx]);
}

void fused_mul_add_cuda(const float *input, const float *alpha, const float *beta, float *output, int n) {
    fused_mul_add_cuda_kernel<<<n / 128, 128>>>(input, alpha, beta, output);
}

int main() {
    const int n = 8192;
    float h_input[n], h_alpha[n], h_beta[n], h_output_naive[n], h_output_fused[n];

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

    CHECK_CUDA(cudaDeviceSynchronize());

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

    return 0;
}