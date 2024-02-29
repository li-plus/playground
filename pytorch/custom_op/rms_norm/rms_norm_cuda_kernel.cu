#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define WARP_SIZE 32

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <int BLOCK_SIZE>
static __device__ __forceinline__ float block_reduce_sum(float x) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    x = warp_reduce_sum(x);
    if constexpr (BLOCK_SIZE > WARP_SIZE) {
        __shared__ float shm[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shm[warp_id] = x;
        }
        __syncthreads();
        x = warp_reduce_sum((lane_id < NUM_WARPS) ? shm[lane_id] : 0.f);
    }
    return x;
}

static __device__ __forceinline__ float2 warp_reduce_sum(float2 v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, mask, 32);
        v.y += __shfl_xor_sync(0xffffffff, v.y, mask, 32);
    }
    return v;
}

template <int BLOCK_SIZE>
static __device__ __forceinline__ float2 block_reduce_sum(float2 v) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    v = warp_reduce_sum(v);
    if constexpr (BLOCK_SIZE > WARP_SIZE) {
        __shared__ float2 shm[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_sum((lane_id < NUM_WARPS) ? shm[lane_id] : make_float2(0.f, 0.f));
    }
    return v;
}

template <int BLOCK_SIZE = 256, int ILP = 4>
static __global__ void rms_norm_cuda_forward_kernel(const float *__restrict__ input, const float *__restrict__ weight,
                                                    float *__restrict__ output, int normalized_shape, float eps) {
    const float *input_row = input + blockIdx.x * normalized_shape;
    float *output_row = output + blockIdx.x * normalized_shape;

    float variances[ILP]{};
    for (int col_start = threadIdx.x; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i++) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const float x = input_row[col];
                variances[i] += x * x;
            }
        }
    }

    float variance = 0.f;
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        variance += variances[i];
    }
    variance = block_reduce_sum<BLOCK_SIZE>(variance);
    const float rrms = rsqrtf(variance / normalized_shape + eps);

    for (int col_start = threadIdx.x; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i++) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                output_row[col] = rrms * input_row[col] * weight[col];
            }
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    torch::Tensor output = torch::empty_like(input);
    const int normalized_shape = input.size(-1);
    const int blocks = input.numel() / normalized_shape;

#define rms_norm_cuda_forward_kernel_launch(BLOCK_SIZE, ILP)                                                           \
    rms_norm_cuda_forward_kernel<BLOCK_SIZE, ILP>                                                                      \
        <<<blocks, BLOCK_SIZE>>>(input.const_data_ptr<float>(), weight.const_data_ptr<float>(),                        \
                                 output.mutable_data_ptr<float>(), normalized_shape, eps)

    if (normalized_shape <= 32) {
        rms_norm_cuda_forward_kernel_launch(32, 1);
    } else if (normalized_shape <= 64) {
        rms_norm_cuda_forward_kernel_launch(32, 2);
    } else if (normalized_shape <= 128) {
        rms_norm_cuda_forward_kernel_launch(32, 4);
    } else if (normalized_shape <= 256) {
        rms_norm_cuda_forward_kernel_launch(64, 4);
    } else if (normalized_shape <= 512) {
        rms_norm_cuda_forward_kernel_launch(128, 4);
    } else if (normalized_shape <= 1024) {
        rms_norm_cuda_forward_kernel_launch(256, 4);
    } else if (normalized_shape <= 2048) {
        rms_norm_cuda_forward_kernel_launch(512, 4);
    } else {
        rms_norm_cuda_forward_kernel_launch(1024, 4);
    }

#undef rms_norm_cuda_forward_kernel_launch

    return output;
}

template <int BLOCK_SIZE = 256, int ILP = 4>
static __global__ void rms_norm_cuda_backward_kernel(const float *grad_output, float *grad_input, float *grad_weight,
                                                     const float *input, const float *weight, int normalized_shape,
                                                     float eps) {
    const float *grad_output_row = grad_output + blockIdx.x * normalized_shape;
    float *grad_input_row = grad_input + blockIdx.x * normalized_shape;
    const float *input_row = input + blockIdx.x * normalized_shape;

    float2 sum_vars[ILP]{};
    for (int col_start = threadIdx.x; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i++) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const float x = input_row[col];
                sum_vars[i].x += x * weight[col] * grad_output_row[col];
                sum_vars[i].y += x * x;
            }
        }
    }

    float2 sum_var = make_float2(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        sum_var.x += sum_vars[i].x;
        sum_var.y += sum_vars[i].y;
    }
    sum_var = block_reduce_sum<BLOCK_SIZE>(sum_var);
    const float sum = sum_var.x;
    const float variance = sum_var.y;

    const float rrms = rsqrtf(variance / normalized_shape + eps);
    const float coef = sum * rrms * rrms * rrms / normalized_shape;

    for (int col_start = threadIdx.x; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i++) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const float x = input_row[col];
                const float grad_output_val = grad_output_row[col];
                grad_input_row[col] = grad_output_val * rrms * weight[col] - coef * x;
                atomicAdd(grad_weight + col, grad_output_val * x * rrms);
            }
        }
    }
}

std::vector<torch::Tensor> rms_norm_cuda_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                                  float eps) {
    torch::Tensor grad_input = torch::empty_like(input);
    torch::Tensor grad_weight = torch::zeros_like(weight);
    const int normalized_shape = input.size(-1);
    const int blocks = input.numel() / normalized_shape;

#define rms_norm_cuda_backward_kernel_launch(BLOCK_SIZE, ILP)                                                          \
    rms_norm_cuda_backward_kernel<BLOCK_SIZE, ILP>                                                                     \
        <<<blocks, BLOCK_SIZE>>>(grad_output.const_data_ptr<float>(), grad_input.mutable_data_ptr<float>(),            \
                                 grad_weight.mutable_data_ptr<float>(), input.const_data_ptr<float>(),                 \
                                 weight.const_data_ptr<float>(), normalized_shape, eps)

    if (normalized_shape <= 32) {
        rms_norm_cuda_backward_kernel_launch(32, 1);
    } else if (normalized_shape <= 64) {
        rms_norm_cuda_backward_kernel_launch(32, 2);
    } else if (normalized_shape <= 128) {
        rms_norm_cuda_backward_kernel_launch(32, 4);
    } else if (normalized_shape <= 256) {
        rms_norm_cuda_backward_kernel_launch(64, 4);
    } else if (normalized_shape <= 512) {
        rms_norm_cuda_backward_kernel_launch(128, 4);
    } else if (normalized_shape <= 1024) {
        rms_norm_cuda_backward_kernel_launch(256, 4);
    } else if (normalized_shape <= 2048) {
        rms_norm_cuda_backward_kernel_launch(512, 4);
    } else {
        rms_norm_cuda_backward_kernel_launch(1024, 4);
    }

#undef rms_norm_cuda_backward_kernel_launch

    return {grad_input, grad_weight};
}
