#include "common.h"

template <int block_size>
__global__ void softmax_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
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

cudaError_t softmax_forward_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 256;
    const int grid_size = M;
    softmax_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    return cudaGetLastError();
}

template <int block_size>
__global__ void softmax_online_forward_kernel(const float *__restrict__ input, float *__restrict__ output, int N) {
    const float *input_row = input + blockIdx.x * N;
    float *output_row = output + blockIdx.x * N;

    float m = -INFINITY;
    float d = 0.f;
    for (int i = threadIdx.x; i < N; i += block_size) {
        const float x = input_row[i];
        const float m_old = m;
        m = fmaxf(m, x);
        d = expf(m_old - m) * d + expf(x - m);
    }
    const float m_local = m;
    m = block_reduce_max<block_size, true>(m_local);
    d = block_reduce_sum<block_size, true>(d * expf(m_local - m));

    const float inv_sum = 1.f / d;
    for (int i = threadIdx.x; i < N; i += block_size) {
        output_row[i] = expf(input_row[i] - m) * inv_sum;
    }
}

cudaError_t softmax_online_forward_cuda(const float *input, float *output, int M, int N) {
    constexpr int block_size = 256;
    const int grid_size = M;
    softmax_online_forward_kernel<block_size><<<grid_size, block_size>>>(input, output, N);
    return cudaGetLastError();
}

cudnnStatus_t softmax_forward_cudnn(cudnnHandle_t handle, const float *input, float *output, int M, int N) {
    cudnnTensorDescriptor_t x_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));

    cudnnTensorDescriptor_t y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));

    const float alpha = 1.f;
    const float beta = 0.f;
    cudnnStatus_t status = cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                                               x_desc, input, &beta, y_desc, output);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));

    return status;
}

template <int block_size>
__global__ void softmax_backward_kernel(const float *__restrict__ grad_output, const float *__restrict__ output,
                                        float *__restrict__ grad_input, int N) {
    const float *grad_output_row = grad_output + blockIdx.x * N;
    const float *output_row = output + blockIdx.x * N;
    float *grad_input_row = grad_input + blockIdx.x * N;

    float y_dot_dy = 0.f;
    for (int i = threadIdx.x; i < N; i += block_size) {
        y_dot_dy += grad_output_row[i] * output_row[i];
    }
    y_dot_dy = block_reduce_sum<block_size, true>(y_dot_dy);

    for (int i = threadIdx.x; i < N; i += block_size) {
        grad_input_row[i] = output_row[i] * (grad_output_row[i] - y_dot_dy);
    }
}

cudaError_t softmax_backward_cuda(const float *grad_output, const float *output, float *grad_input, int M, int N) {
    constexpr int block_size = 256;
    const int grid_size = M;
    softmax_backward_kernel<block_size><<<grid_size, block_size>>>(grad_output, output, grad_input, N);
    return cudaGetLastError();
}

cudnnStatus_t softmax_backward_cudnn(cudnnHandle_t handle, const float *grad_output, const float *output,
                                     float *grad_input, int M, int N) {
    cudnnTensorDescriptor_t y_desc, dy_desc, dx_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dy_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dx_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, M, N, 1, 1));

    const float alpha = 1.f;
    const float beta = 0.f;
    cudnnStatus_t status = cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
                                                y_desc, output, dy_desc, grad_output, &beta, dx_desc, grad_input);

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(dy_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(dx_desc));

    return status;
}

void run_softmax_forward(cudnnHandle_t handle, int M, int N) {
    float *h_x, *h_y_expect, *h_y_actual;
    CHECK_CUDA(cudaMallocHost(&h_x, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_expect, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_y_actual, M * N * sizeof(float), cudaHostAllocDefault));

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, M * N * sizeof(float)));

    // initialize x
    for (int i = 0; i < M * N; i++) {
        h_x[i] = uniform();
    }
    CHECK_CUDA(cudaMemcpy(d_x, h_x, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuda forward
    CHECK_CUDA(cudaMemsetAsync(d_y, 0, M * N * sizeof(float)));
    CHECK_CUDA(softmax_forward_cuda(d_x, d_y, M, N));
    CHECK_CUDA(cudaMemcpy(h_y_expect, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // online forward
    CHECK_CUDA(cudaMemsetAsync(d_y, 0, M * N * sizeof(float)));
    CHECK_CUDA(softmax_online_forward_cuda(d_x, d_y, M, N));
    CHECK_CUDA(cudaMemcpy(h_y_actual, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_y_expect, h_y_actual, M * N);

    // cudnn forward
    CHECK_CUDA(cudaMemsetAsync(d_y, 0, M * N * sizeof(float)));
    CHECK_CUDNN(softmax_forward_cudnn(handle, d_x, d_y, M, N));
    CHECK_CUDA(cudaMemcpy(h_y_actual, d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_y_expect, h_y_actual, M * N);

    // benchmark forward
    printf("===== forward =====\n");
    {
        const float elapsed = timeit([&] { CHECK_CUDA(softmax_forward_cuda(d_x, d_y, M, N)); }, 10, 100);
        printf("[cuda] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed = timeit([&] { CHECK_CUDA(softmax_online_forward_cuda(d_x, d_y, M, N)); }, 10, 100);
        printf("[online] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed = timeit([&] { CHECK_CUDNN(softmax_forward_cudnn(handle, d_x, d_y, M, N)); }, 10, 100);
        printf("[cudnn] elapsed %.3f us\n", elapsed * 1e6);
    }
    // [cuda] elapsed 10.383 us
    // [online] elapsed 11.233 us
    // [cudnn] elapsed 13.445 us

    // clean up
    CHECK_CUDA(cudaFreeHost(h_x));
    CHECK_CUDA(cudaFreeHost(h_y_expect));
    CHECK_CUDA(cudaFreeHost(h_y_actual));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
}

void run_softmax_backward(cudnnHandle_t handle, int M, int N) {
    float *h_y, *h_dy, *h_dx_expect, *h_dx_actual;
    CHECK_CUDA(cudaMallocHost(&h_y, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_dy, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_dx_expect, M * N * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_dx_actual, M * N * sizeof(float), cudaHostAllocDefault));

    float *d_y, *d_dy, *d_dx;
    CHECK_CUDA(cudaMalloc(&d_y, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dy, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dx, M * N * sizeof(float)));

    // initialize
    for (int i = 0; i < M * N; i++) {
        h_y[i] = uniform(-1.0, 1.0);
        h_dy[i] = uniform(-1.0, 1.0);
    }
    CHECK_CUDA(cudaMemcpyAsync(d_y, h_y, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyAsync(d_dy, h_dy, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuda backward
    CHECK_CUDA(cudaMemsetAsync(d_dx, 0, M * N * sizeof(float)));
    CHECK_CUDA(softmax_backward_cuda(d_dy, d_y, d_dx, M, N));
    CHECK_CUDA(cudaMemcpy(h_dx_expect, d_dx, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cudnn backward
    CHECK_CUDA(cudaMemsetAsync(d_dx, 0, M * N * sizeof(float)));
    CHECK_CUDNN(softmax_backward_cudnn(handle, d_dy, d_y, d_dx, M, N));
    CHECK_CUDA(cudaMemcpy(h_dx_actual, d_dx, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    check_is_close(h_dx_expect, h_dx_actual, M * N);

    // benchmark backward
    printf("===== backward =====\n");
    {
        const float elapsed = timeit([&] { CHECK_CUDA(softmax_backward_cuda(d_dy, d_y, d_dx, M, N)); }, 10, 100);
        printf("[cuda] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed =
            timeit([&] { CHECK_CUDNN(softmax_backward_cudnn(handle, d_dy, d_y, d_dx, M, N)); }, 10, 100);
        printf("[cudnn] elapsed %.3f us\n", elapsed * 1e6);
    }
    // [cuda] elapsed 10.168 us
    // [cudnn] elapsed 12.595 us

    // clean up
    CHECK_CUDA(cudaFreeHost(h_y));
    CHECK_CUDA(cudaFreeHost(h_dy));
    CHECK_CUDA(cudaFreeHost(h_dx_expect));
    CHECK_CUDA(cudaFreeHost(h_dx_actual));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_dy));
    CHECK_CUDA(cudaFree(d_dx));
}

int main() {
    constexpr int M = 1024;
    constexpr int N = 2048;

    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    run_softmax_forward(handle, M, N);
    run_softmax_backward(handle, M, N);

    CHECK_CUDNN(cudnnDestroy(handle));

    return 0;
}