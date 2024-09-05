#include "common.h"
#include <cudnn.h>

#define CHECK_CUDNN(call)                                                                                              \
    do {                                                                                                               \
        cudnnStatus_t status = (call);                                                                                 \
        CHECK(status == CUDNN_STATUS_SUCCESS) << "cudnn error: " << cudnnGetErrorString(status);                       \
    } while (false)

template <int block_size>
__global__ void softmax_kernel(const float *input, float *output, int N) {
    const float *input_row = input + blockIdx.x * N;
    float *output_row = output + blockIdx.x * N;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += block_size) {
        max_val = fmaxf(max_val, input_row[i]);
    }
    max_val = block_reduce_max<block_size, true>(max_val);

    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += block_size) {
        sum += expf(input_row[i] - max_val);
    }
    sum = block_reduce_sum<block_size, true>(sum);

    const float inv_sum = 1.f / sum;
    for (int i = threadIdx.x; i < N; i += block_size) {
        output_row[i] = expf(input_row[i] - max_val) * inv_sum;
    }
}

void softmax_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 256;
    const int grid_size = M;
    softmax_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    CHECK_CUDA(cudaGetLastError());
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 2048;

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    cudnnTensorDescriptor_t x_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));

    cudnnTensorDescriptor_t y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));

    float *h_x, *h_y_cuda, *h_y_cudnn;
    CHECK_CUDA(cudaMallocHost(&h_x, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_cuda, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_cudnn, M * N * sizeof(float), cudaHostAllocDefault));

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, M * N * sizeof(float)));

    // initialize x
    for (int i = 0; i < M * N; i++) {
        h_x[i] = uniform();
    }
    CHECK_CUDA(cudaMemcpy(d_x, h_x, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuda
    CHECK_CUDA(cudaMemsetAsync(d_y, 0, M * N * sizeof(float)));
    softmax_cuda(d_x, d_y, M, N);
    CHECK_CUDA(cudaMemcpy(h_y_cuda, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cudnn
    const float alpha = 1.f;
    const float beta = 0.f;
    CHECK_CUDA(cudaMemsetAsync(d_y, 0, M * N * sizeof(float)));
    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, x_desc, d_x, &beta, y_desc,
                        d_y);
    CHECK_CUDA(cudaMemcpy(h_y_cudnn, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    check_is_close(h_y_cuda, h_y_cudnn, M * N, 1e-4f);

    // benchmark
    {
        const float elapsed = timeit([&] { softmax_cuda(d_x, d_y, M, N); }, 10, 100);
        printf("[cuda] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed = timeit(
            [&] {
                cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, x_desc, d_x,
                                    &beta, y_desc, d_y);
            },
            10, 100);
        printf("[cudnn] elapsed %.3f us\n", elapsed * 1e6);
    }

    // clean up
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));

    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_y_cuda));
    CHECK_CUDA(cudaFreeHost(h_y_cudnn));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    CHECK_CUDNN(cudnnDestroy(handle));

    return 0;
}