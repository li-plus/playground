#include "common.h"

__global__ void add_f1_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                   float *__restrict__ output, int N) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        output[i] = input[i] + other[i];
    }
}

static void add_f1_cuda(const float *input, const float *other, float *output, int N) {
    add_f1_cuda_kernel<<<N / 1024 / 8, 1024>>>(input, other, output, N);
}

__global__ void add_f4_cuda_kernel(const float *__restrict__ input, const float *__restrict__ other,
                                   float *__restrict__ output, int N) {
    for (int i = (blockDim.x * blockIdx.x + threadIdx.x) * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        float4 a4 = *(float4 *)&input[i];
        float4 b4 = *(float4 *)&other[i];
        *(float4 *)&output[i] = make_float4(a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    }
}

static void add_f4_cuda(const float *input, const float *other, float *output, int N) {
    add_f4_cuda_kernel<<<N / 1024 / 8, 1024>>>(input, other, output, N);
}

int main() {
    constexpr size_t N = 128ull * 1024 * 1024;

    float *h_input, *h_other, *h_output_f1, *h_output_f4;
    CHECK_CUDA(cudaMallocHost(&h_input, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_other, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_output_f1, N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_output_f4, N * sizeof(float), cudaHostAllocDefault));

    float *d_input, *d_other, *d_output_f1, *d_output_f4;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_other, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_f1, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_f4, N * sizeof(float)));

    // set inputs
    for (int i = 0; i < N; i++) {
        h_input[i] = i;
        h_other[i] = 1;
    }

    CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_other, h_other, N * sizeof(float), cudaMemcpyHostToDevice));

    // compute
    add_f1_cuda(d_input, d_other, d_output_f1, N);
    CHECK_CUDA(cudaMemcpy(h_output_f1, d_output_f1, N * sizeof(float), cudaMemcpyDeviceToHost));

    add_f4_cuda(d_input, d_other, d_output_f4, N);
    CHECK_CUDA(cudaMemcpy(h_output_f4, d_output_f4, N * sizeof(float), cudaMemcpyDeviceToHost));

    // check results
    check_is_close(h_output_f1, h_output_f4, N);

    // benchmark
    const float elapsed_f1 = timeit([=] { add_f1_cuda(d_input, d_other, d_output_f1, N); }, 2, 10);
    const float elapsed_f4 = timeit([=] { add_f4_cuda(d_input, d_other, d_output_f4, N); }, 2, 10);
    const float bandwidth_f1 = 3 * N * sizeof(float) / 1e9 / elapsed_f1;
    const float bandwidth_f4 = 3 * N * sizeof(float) / 1e9 / elapsed_f4;
    printf("[float1] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_f1 * 1e6, bandwidth_f1);
    printf("[float4] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed_f4 * 1e6, bandwidth_f4);

    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_other));
    CHECK_CUDA(cudaFreeHost(h_output_f1));
    CHECK_CUDA(cudaFreeHost(h_output_f4));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_other));
    CHECK_CUDA(cudaFree(d_output_f1));
    CHECK_CUDA(cudaFree(d_output_f4));

    return 0;
}
