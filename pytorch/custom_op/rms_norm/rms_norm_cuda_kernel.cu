#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE 32

// Copied from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/static_switch.h
#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                             \
    [&] {                                                                                                              \
        if (COND) {                                                                                                    \
            constexpr static bool CONST_NAME = true;                                                                   \
            return __VA_ARGS__();                                                                                      \
        } else {                                                                                                       \
            constexpr static bool CONST_NAME = false;                                                                  \
            return __VA_ARGS__();                                                                                      \
        }                                                                                                              \
    }()

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

// float4 utils
static __device__ __forceinline__ float4 operator*(const float4 &self, float other) {
    return make_float4(self.x * other, self.y * other, self.z * other, self.w * other);
}

static __device__ __forceinline__ float4 operator*(float self, const float4 &other) { return other * self; }

static __device__ __forceinline__ float4 operator*(const float4 &self, const float4 &other) {
    return make_float4(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w);
}

static __device__ __forceinline__ float4 operator+(const float4 &self, const float4 &other) {
    return make_float4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w);
}

static __device__ __forceinline__ float4 operator+=(float4 &self, const float4 &other) { return self = self + other; }

static __device__ __forceinline__ float4 operator-(const float4 &self, const float4 &other) {
    return make_float4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w);
}

// float2 utils
static __device__ __forceinline__ float2 operator*(const float2 &self, float other) {
    return make_float2(self.x * other, self.y * other);
}

static __device__ __forceinline__ float2 operator*(float self, const float2 &other) { return other * self; }

static __device__ __forceinline__ float2 operator*(const float2 &self, const float2 &other) {
    return make_float2(self.x * other.x, self.y * other.y);
}

static __device__ __forceinline__ float2 operator+(const float2 &self, const float2 &other) {
    return make_float2(self.x + other.x, self.y + other.y);
}

static __device__ __forceinline__ float2 operator+=(float2 &self, const float2 &other) { return self = self + other; }

static __device__ __forceinline__ float2 operator-(const float2 &self, const float2 &other) {
    return make_float2(self.x - other.x, self.y - other.y);
}

template <int BLOCK_SIZE = 256, int ILP = 4, int FLOAT_ALIGN = 1>
static __global__ void rms_norm_cuda_forward_kernel(const float *__restrict__ input, const float *__restrict__ weight,
                                                    float *__restrict__ output, int normalized_shape, float eps) {
    const float *input_row = input + blockIdx.x * normalized_shape;
    float *output_row = output + blockIdx.x * normalized_shape;

    constexpr int N = std::min(FLOAT_ALIGN, ILP);
    static_assert(ILP % N == 0);
    using floatN = std::conditional_t<N == 4, float4, std::conditional_t<N == 2, float2, float>>;

    float variances[ILP]{};
    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const floatN x = *(floatN *)&input_row[col];
                *(floatN *)&variances[i] += x * x;
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

    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                *(floatN *)&output_row[col] = rrms * *(floatN *)&input_row[col] * *(floatN *)&weight[col];
            }
        }
    }
}

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    torch::Tensor output = torch::empty_like(input);
    const int normalized_shape = input.size(-1);
    const int blocks = input.numel() / normalized_shape;

#define rms_norm_cuda_forward_kernel_launch(BLOCK_SIZE, ILP, FLOAT_ALIGN)                                              \
    rms_norm_cuda_forward_kernel<BLOCK_SIZE, ILP, FLOAT_ALIGN>                                                         \
        <<<blocks, BLOCK_SIZE>>>(input.const_data_ptr<float>(), weight.const_data_ptr<float>(),                        \
                                 output.mutable_data_ptr<float>(), normalized_shape, eps)

    BOOL_SWITCH(normalized_shape % 4 == 0, FLOAT_ALIGN4, [&] {
        BOOL_SWITCH(normalized_shape % 2 == 0, FLOAT_ALIGN2, [&] {
            constexpr int FLOAT_ALIGN = FLOAT_ALIGN4 ? 4 : (FLOAT_ALIGN2 ? 2 : 1);
            if (normalized_shape <= 32) {
                rms_norm_cuda_forward_kernel_launch(32, 1, FLOAT_ALIGN);
            } else if (normalized_shape <= 64) {
                rms_norm_cuda_forward_kernel_launch(32, 2, FLOAT_ALIGN);
            } else if (normalized_shape <= 128) {
                rms_norm_cuda_forward_kernel_launch(32, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 256) {
                rms_norm_cuda_forward_kernel_launch(64, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 512) {
                rms_norm_cuda_forward_kernel_launch(128, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 1024) {
                rms_norm_cuda_forward_kernel_launch(256, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 2048) {
                rms_norm_cuda_forward_kernel_launch(512, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 4096) {
                rms_norm_cuda_forward_kernel_launch(1024, 4, FLOAT_ALIGN);
            } else {
                rms_norm_cuda_forward_kernel_launch(1024, 8, FLOAT_ALIGN);
            }
        });
    });

#undef rms_norm_cuda_forward_kernel_launch

    return output;
}

template <int BLOCK_SIZE = 256, int ILP = 4, int FLOAT_ALIGN = 1>
static __global__ void
rms_norm_cuda_backward_kernel(const float *__restrict__ grad_output, float *__restrict__ grad_input,
                              float *__restrict__ grad_weight_partial, const float *__restrict__ input,
                              const float *__restrict__ weight, int normalized_shape, float eps) {
    const float *grad_output_row = grad_output + blockIdx.x * normalized_shape;
    float *grad_input_row = grad_input + blockIdx.x * normalized_shape;
    float *grad_weight_partial_row = grad_weight_partial + blockIdx.x * normalized_shape;
    const float *input_row = input + blockIdx.x * normalized_shape;

    constexpr int N = std::min(FLOAT_ALIGN, ILP);
    static_assert(ILP % N == 0);
    using floatN = std::conditional_t<N == 4, float4, std::conditional_t<N == 2, float2, float>>;

    float sums[ILP]{};
    float vars[ILP]{};
    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const floatN x = *(floatN *)&input_row[col];
                *(floatN *)&sums[i] += x * *(floatN *)&weight[col] * *(floatN *)&grad_output_row[col];
                *(floatN *)&vars[i] += x * x;
            }
        }
    }

    float2 sum_var = make_float2(0.f, 0.f);
#pragma unroll
    for (int i = 0; i < ILP; i++) {
        sum_var.x += sums[i];
        sum_var.y += vars[i];
    }
    sum_var = block_reduce_sum<BLOCK_SIZE>(sum_var);
    const float sum = sum_var.x;
    const float variance = sum_var.y;

    const float rrms = rsqrtf(variance / normalized_shape + eps);
    const float coef = (sum * rrms) * (rrms * rrms) / normalized_shape;

    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const floatN x = *(floatN *)&input_row[col];
                const floatN grad_output_rrms = *(floatN *)&grad_output_row[col] * rrms;
                *(floatN *)&grad_input_row[col] = grad_output_rrms * *(floatN *)&weight[col] - coef * x;
                *(floatN *)&grad_weight_partial_row[col] = grad_output_rrms * x;
            }
        }
    }
}

std::vector<torch::Tensor> rms_norm_cuda_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                                  float eps) {
    torch::Tensor grad_input = torch::empty_like(input);
    torch::Tensor grad_weight_partial = torch::empty_like(input);
    const int normalized_shape = input.size(-1);
    const int blocks = input.numel() / normalized_shape;

#define rms_norm_cuda_backward_kernel_launch(BLOCK_SIZE, ILP, FLOAT_ALIGN)                                             \
    rms_norm_cuda_backward_kernel<BLOCK_SIZE, ILP, FLOAT_ALIGN><<<blocks, BLOCK_SIZE, FLOAT_ALIGN>>>(                  \
        grad_output.const_data_ptr<float>(), grad_input.mutable_data_ptr<float>(),                                     \
        grad_weight_partial.mutable_data_ptr<float>(), input.const_data_ptr<float>(), weight.const_data_ptr<float>(),  \
        normalized_shape, eps)

    BOOL_SWITCH(normalized_shape % 4 == 0, FLOAT_ALIGN4, [&] {
        BOOL_SWITCH(normalized_shape % 2 == 0, FLOAT_ALIGN2, [&] {
            constexpr int FLOAT_ALIGN = FLOAT_ALIGN4 ? 4 : (FLOAT_ALIGN2 ? 2 : 1);
            if (normalized_shape <= 32) {
                rms_norm_cuda_backward_kernel_launch(32, 1, FLOAT_ALIGN);
            } else if (normalized_shape <= 64) {
                rms_norm_cuda_backward_kernel_launch(32, 2, FLOAT_ALIGN);
            } else if (normalized_shape <= 128) {
                rms_norm_cuda_backward_kernel_launch(32, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 256) {
                rms_norm_cuda_backward_kernel_launch(64, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 512) {
                rms_norm_cuda_backward_kernel_launch(128, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 1024) {
                rms_norm_cuda_backward_kernel_launch(256, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 2048) {
                rms_norm_cuda_backward_kernel_launch(512, 4, FLOAT_ALIGN);
            } else if (normalized_shape <= 4096) {
                rms_norm_cuda_backward_kernel_launch(1024, 4, FLOAT_ALIGN);
            } else {
                rms_norm_cuda_backward_kernel_launch(1024, 8, FLOAT_ALIGN);
            }
        });
    });

#undef rms_norm_cuda_backward_kernel_launch

    torch::Tensor grad_weight = grad_weight_partial.view({-1, normalized_shape}).sum(0);

    return {grad_input, grad_weight};
}
