#include "common.h"

// coalesce memory access
__global__ void coalesce_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                     float *__restrict__ output, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] + other[i];
    }
}

void coalesce_cuda(const float *input, const float *other, float *output, int n) {
    coalesce_cuda_kernel<<<1, 128>>>(input, other, output, n);
}

// non-coalesce memory access
__global__ void non_coalesce_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                         float *__restrict__ output, int chunk_size) {
    for (int i = threadIdx.x * chunk_size; i < (threadIdx.x + 1) * chunk_size; i++) {
        output[i] = input[i] + other[i];
    }
}

void non_coalesce_cuda(const float *input, const float *other, float *output, int n) {
    non_coalesce_cuda_kernel<<<1, 128>>>(input, other, output, n / 128);
}

int main() {
    const int n = 1024 * 64;

    float *h_input, *h_other, *h_output_non, *h_output_coa;

    CHECK_CUDA(cudaMallocHost(&h_input, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_other, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_non, n * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output_coa, n * sizeof(float)));

    float *d_input, *d_other, *d_output_non, *d_output_coa;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_other, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_non, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_coa, n * sizeof(float)));

    for (int i = 0; i < n; i++) {
        h_input[i] = uniform();
        h_other[i] = uniform();
    }

    CHECK_CUDA(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_other, h_other, n * sizeof(float), cudaMemcpyHostToDevice));

    non_coalesce_cuda(d_input, d_other, d_output_non, n);
    CHECK_CUDA(cudaMemcpy(h_output_non, d_output_non, n * sizeof(float), cudaMemcpyDeviceToHost));

    coalesce_cuda(d_input, d_other, d_output_coa, n);
    CHECK_CUDA(cudaMemcpy(h_output_coa, d_output_coa, n * sizeof(float), cudaMemcpyDeviceToHost));

    // check correctness
    for (int i = 0; i < n; i++) {
        CHECK(is_close(h_output_non[i], h_output_coa[i]));
    }

    const float non_elapsed = timeit([=] { non_coalesce_cuda(d_input, d_other, d_output_non, n); }, 10, 1000);
    const float coa_elapsed = timeit([=] { coalesce_cuda(d_input, d_other, d_output_coa, n); }, 10, 1000);

    printf("[coalesce]:     %.3f us\n", coa_elapsed * 1e6f);
    printf("[non-coalesce]: %.3f us\n", non_elapsed * 1e6f);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_other));
    CHECK_CUDA(cudaFree(d_output_coa));
    CHECK_CUDA(cudaFree(d_output_non));

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_other));
    CHECK_CUDA(cudaFreeHost(h_output_coa));
    CHECK_CUDA(cudaFreeHost(h_output_non));

    return 0;
}