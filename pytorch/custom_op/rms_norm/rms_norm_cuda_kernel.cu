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

__device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float x) {
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

__device__ __forceinline__ float2 warp_reduce_sum(float2 v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, mask, 32);
        v.y += __shfl_xor_sync(0xffffffff, v.y, mask, 32);
    }
    return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float2 block_reduce_sum(float2 v) {
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

// float4 operation
__device__ __forceinline__ float4 operator*(const float4 &self, float other) {
    return make_float4(self.x * other, self.y * other, self.z * other, self.w * other);
}

__device__ __forceinline__ float4 operator*(float self, const float4 &other) { return other * self; }

__device__ __forceinline__ float4 operator*(const float4 &self, const float4 &other) {
    return make_float4(self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w);
}

__device__ __forceinline__ float4 operator+(const float4 &self, const float4 &other) {
    return make_float4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w);
}

__device__ __forceinline__ float4 operator+=(float4 &self, const float4 &other) { return self = self + other; }

__device__ __forceinline__ float4 operator-(const float4 &self, const float4 &other) {
    return make_float4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w);
}

// load from global memory
template <typename Dst, typename Src>
__device__ __forceinline__ Dst load_as(const Src *ptr) {
    static_assert(std::is_same_v<Dst, float> || std::is_same_v<Dst, float4>, "unsupported dtype");
    static_assert(std::is_same_v<Src, double> || std::is_same_v<Src, float> || std::is_same_v<Src, half> ||
                      std::is_same_v<Src, nv_bfloat16>,
                  "unsupported dtype");

    if constexpr (std::is_same_v<Dst, float>) {
        if constexpr (std::is_same_v<Src, half>) {
            return __half2float(*ptr);
        } else if constexpr (std::is_same_v<Src, nv_bfloat16>) {
            return __bfloat162float(*ptr);
        } else {
            return *ptr;
        }
    } else {
        static_assert(std::is_same_v<Dst, float4>);
        static_assert(!std::is_same_v<Src, double>, "bad performance");
        if constexpr (std::is_same_v<Src, float>) {
            return *(float4 *)ptr;
        } else if constexpr (std::is_same_v<Src, half>) {
            half2 h2[2];
            *(float2 *)h2 = *(float2 *)ptr;
            float2 f01 = __half22float2(h2[0]);
            float2 f23 = __half22float2(h2[1]);
            return make_float4(f01.x, f01.y, f23.x, f23.y);
        } else if constexpr (std::is_same_v<Src, nv_bfloat16>) {
            nv_bfloat162 bf2[2];
            *(float2 *)bf2 = *(float2 *)ptr;
            float2 f01 = __bfloat1622float2(bf2[0]);
            float2 f23 = __bfloat1622float2(bf2[1]);
            return make_float4(f01.x, f01.y, f23.x, f23.y);
        }
    }
}

// store to global memory
template <typename Dst, typename Src>
__device__ __forceinline__ void store_as(Dst *ptr, Src val) {
    static_assert(std::is_same_v<Src, float> || std::is_same_v<Src, float4>, "unsupported dtype");
    static_assert(std::is_same_v<Dst, double> || std::is_same_v<Dst, float> || std::is_same_v<Dst, half> ||
                      std::is_same_v<Dst, nv_bfloat16>,
                  "unsupported dtype");

    if constexpr (std::is_same_v<Src, float>) {
        if constexpr (std::is_same_v<Dst, half>) {
            *ptr = __float2half(val);
        } else if constexpr (std::is_same_v<Dst, nv_bfloat16>) {
            *ptr = __float2bfloat16(val);
        } else {
            *ptr = val;
        }
    } else {
        static_assert(std::is_same_v<Src, float4>);
        static_assert(!std::is_same_v<Dst, double>, "bad performance");
        if constexpr (std::is_same_v<Dst, float>) {
            *(float4 *)ptr = val;
        } else if constexpr (std::is_same_v<Dst, half>) {
            half2 h2[2]{__float22half2_rn(*(float2 *)&val.x), __float22half2_rn(*(float2 *)&val.z)};
            *(float2 *)ptr = *(float2 *)h2;
        } else {
            static_assert(std::is_same_v<Dst, nv_bfloat16>);
            nv_bfloat162 bf2[2]{__float22bfloat162_rn(*(float2 *)&val.x), __float22bfloat162_rn(*(float2 *)&val.z)};
            *(float2 *)ptr = *(float2 *)bf2;
        }
    }
}

template <typename scalar_t, int BLOCK_SIZE = 256, int ILP = 4, int ALIGN = 1>
__global__ void rms_norm_cuda_forward_kernel(const scalar_t *__restrict__ input, const scalar_t *__restrict__ weight,
                                             scalar_t *__restrict__ output, int normalized_shape, float eps) {
    static_assert(std::is_same_v<scalar_t, double> || std::is_same_v<scalar_t, float> ||
                      std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, nv_bfloat16>,
                  "unsupported dtype");
    const scalar_t *input_row = input + blockIdx.x * normalized_shape;
    scalar_t *output_row = output + blockIdx.x * normalized_shape;

    constexpr int N = (std::min(ALIGN, ILP) == 4) ? 4 : 1;
    static_assert(ILP % N == 0);
    using floatN = std::conditional_t<N == 4, float4, float>;

    float variances[ILP]{};
    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const floatN x = load_as<floatN>(&input_row[col]);
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
                store_as(&output_row[col], rrms * load_as<floatN>(&input_row[col]) * load_as<floatN>(&weight[col]));
            }
        }
    }
}

// Convert torch::Half and torch::BFloat16 to half and nv_bfloat16, respectively
template <typename torch_scalar_t>
struct cuda_scalar {
    using type = torch_scalar_t;
};

template <>
struct cuda_scalar<torch::Half> {
    using type = half;
};

template <>
struct cuda_scalar<torch::BFloat16> {
    using type = nv_bfloat16;
};

template <typename torch_scalar_t>
using cuda_scalar_t = typename cuda_scalar<torch_scalar_t>::type;

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    torch::Tensor output = torch::empty_like(input);
    const int normalized_shape = input.size(-1);
    const int blocks = input.numel() / normalized_shape;

#define rms_norm_cuda_forward_kernel_launch(scalar_t, BLOCK_SIZE, ILP, ALIGN)                                          \
    rms_norm_cuda_forward_kernel<cuda_scalar_t<scalar_t>, BLOCK_SIZE, ILP, ALIGN><<<blocks, BLOCK_SIZE>>>(             \
        (const cuda_scalar_t<scalar_t> *)input.const_data_ptr<scalar_t>(),                                             \
        (const cuda_scalar_t<scalar_t> *)weight.const_data_ptr<scalar_t>(),                                            \
        (cuda_scalar_t<scalar_t> *)output.mutable_data_ptr<scalar_t>(), normalized_shape, eps)

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, input.scalar_type(),
                                    "rms_norm_cuda_forward", [&] {
                                        BOOL_SWITCH(normalized_shape % 4 == 0, ALIGN4, [&] {
                                            constexpr int ALIGN = (ALIGN4 && !std::is_same_v<scalar_t, double>) ? 4 : 1;
                                            if (normalized_shape <= 32) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 32, 1, ALIGN);
                                            } else if (normalized_shape <= 64) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 32, 2, ALIGN);
                                            } else if (normalized_shape <= 128) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 32, 4, ALIGN);
                                            } else if (normalized_shape <= 256) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 64, 4, ALIGN);
                                            } else if (normalized_shape <= 512) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 128, 4, ALIGN);
                                            } else if (normalized_shape <= 1024) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 256, 4, ALIGN);
                                            } else if (normalized_shape <= 2048) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 512, 4, ALIGN);
                                            } else if (normalized_shape <= 4096) {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 1024, 4, ALIGN);
                                            } else {
                                                rms_norm_cuda_forward_kernel_launch(scalar_t, 1024, 8, ALIGN);
                                            }
                                        });
                                    });

#undef rms_norm_cuda_forward_kernel_launch

    return output;
}

template <typename scalar_t, int BLOCK_SIZE = 256, int ILP = 4, int ALIGN = 1>
__global__ void
rms_norm_cuda_backward_kernel(const scalar_t *__restrict__ grad_output, scalar_t *__restrict__ grad_input,
                              scalar_t *__restrict__ grad_weight_partial, const scalar_t *__restrict__ input,
                              const scalar_t *__restrict__ weight, int normalized_shape, float eps) {
    const scalar_t *grad_output_row = grad_output + blockIdx.x * normalized_shape;
    scalar_t *grad_input_row = grad_input + blockIdx.x * normalized_shape;
    scalar_t *grad_weight_partial_row = grad_weight_partial + blockIdx.x * normalized_shape;
    const scalar_t *input_row = input + blockIdx.x * normalized_shape;

    constexpr int N = (std::min(ALIGN, ILP) == 4) ? 4 : 1;
    static_assert(ILP % N == 0);
    using floatN = std::conditional_t<N == 4, float4, float>;

    float sums[ILP]{};
    float vars[ILP]{};
    for (int col_start = threadIdx.x * N; col_start < normalized_shape; col_start += BLOCK_SIZE * ILP) {
#pragma unroll
        for (int i = 0; i < ILP; i += N) {
            const int col = col_start + i * BLOCK_SIZE;
            if (col < normalized_shape) {
                const floatN x = load_as<floatN>(&input_row[col]);
                *(floatN *)&sums[i] += x * load_as<floatN>(&weight[col]) * load_as<floatN>(&grad_output_row[col]);
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
                const floatN x = load_as<floatN>(&input_row[col]);
                const floatN grad_output_rrms = load_as<floatN>(&grad_output_row[col]) * rrms;
                store_as(&grad_input_row[col], grad_output_rrms * load_as<floatN>(&weight[col]) - coef * x);
                store_as(&grad_weight_partial_row[col], grad_output_rrms * x);
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

#define rms_norm_cuda_backward_kernel_launch(scalar_t, BLOCK_SIZE, ILP, FLOAT_ALIGN)                                   \
    rms_norm_cuda_backward_kernel<cuda_scalar_t<scalar_t>, BLOCK_SIZE, ILP, FLOAT_ALIGN>                               \
        <<<blocks, BLOCK_SIZE, FLOAT_ALIGN>>>(                                                                         \
            (const cuda_scalar_t<scalar_t> *)grad_output.const_data_ptr<scalar_t>(),                                   \
            (cuda_scalar_t<scalar_t> *)grad_input.mutable_data_ptr<scalar_t>(),                                        \
            (cuda_scalar_t<scalar_t> *)grad_weight_partial.mutable_data_ptr<scalar_t>(),                               \
            (const cuda_scalar_t<scalar_t> *)input.const_data_ptr<scalar_t>(),                                         \
            (const cuda_scalar_t<scalar_t> *)weight.const_data_ptr<scalar_t>(), normalized_shape, eps)

    AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half, torch::ScalarType::BFloat16, input.scalar_type(),
                                    "rms_norm_cuda_backward", [&] {
                                        BOOL_SWITCH(normalized_shape % 4 == 0, ALIGN4, [&] {
                                            constexpr int ALIGN = (ALIGN4 && !std::is_same_v<scalar_t, double>) ? 4 : 1;
                                            if (normalized_shape <= 32) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 32, 1, ALIGN);
                                            } else if (normalized_shape <= 64) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 32, 2, ALIGN);
                                            } else if (normalized_shape <= 128) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 32, 4, ALIGN);
                                            } else if (normalized_shape <= 256) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 64, 4, ALIGN);
                                            } else if (normalized_shape <= 512) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 128, 4, ALIGN);
                                            } else if (normalized_shape <= 1024) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 256, 4, ALIGN);
                                            } else if (normalized_shape <= 2048) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 512, 4, ALIGN);
                                            } else if (normalized_shape <= 4096) {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 1024, 4, ALIGN);
                                            } else {
                                                rms_norm_cuda_backward_kernel_launch(scalar_t, 1024, 8, ALIGN);
                                            }
                                        });
                                    });

#undef rms_norm_cuda_backward_kernel_launch

    torch::Tensor grad_weight = grad_weight_partial.view({-1, normalized_shape}).sum(0);

    return {grad_input, grad_weight};
}
