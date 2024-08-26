#include <cstdint>
#include <cuda_fp16.h>

template <int warp_size = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, mask, 32);
    }
    return v;
}

template <int block_size, bool all>
__device__ __forceinline__ float block_reduce_sum(float v) {
    constexpr int warp_size = 32;
    static_assert(block_size % warp_size == 0, "invalid block size");
    v = warp_reduce_sum(v);
    if constexpr (block_size > warp_size) {
        constexpr int num_warps = block_size / warp_size;
        __shared__ float shm[num_warps];
        const int warp_id = threadIdx.x / warp_size;
        const int lane_id = threadIdx.x % warp_size;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        constexpr int warp_reduce_size = all ? (1024 / warp_size) : num_warps;
        v = warp_reduce_sum<warp_reduce_size>((lane_id < num_warps) ? shm[lane_id] : 0.f);
    }
    return v;
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases.
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu
__device__ __forceinline__ __half2 dequant_w8(int q) {
    constexpr int LO = 0x00ff00ff;
    constexpr int EX = 0x64006400;
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    constexpr int SUB = 0x64806480;
    __half2 result = __hsub2(*(__half2 *)&lo, *(__half2 *)&SUB);
    return result;
}

template <int group_size, int block_size>
__global__ void gemv_w8_kernel(const __half *__restrict__ input, const uint8_t *__restrict__ weight,
                               const __half *__restrict__ scales, const __half *__restrict__ bias,
                               __half *__restrict__ output, int N) {
    static_assert(group_size % 16 == 0, "invalid group_size");

    float sum = 0.f;

    const __half *scales_row = scales + blockIdx.x * (N / group_size);
    const uint8_t *weight_row = weight + blockIdx.x * N;

    for (int i = threadIdx.x * 16; i < N; i += block_size * 16) {
        int4 wq = *(int4 *)&weight_row[i];

        float partial_sum = 0.f;

        __half2 w[4];
        __half2 x[4];

        *(float4 *)x = *(float4 *)&input[i + 0];

        w[0] = dequant_w8(wq.x);
        w[1] = dequant_w8(wq.x >> 8);
        w[2] = dequant_w8(wq.y);
        w[3] = dequant_w8(wq.y >> 8);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 prod = __half22float2(__hmul2(w[j], x[j]));
            partial_sum += prod.x + prod.y;
        }

        *(float4 *)x = *(float4 *)&input[i + 8];

        w[0] = dequant_w8(wq.z);
        w[1] = dequant_w8(wq.z >> 8);
        w[2] = dequant_w8(wq.w);
        w[3] = dequant_w8(wq.w >> 8);

        const float scale = __half2float(scales_row[i / group_size]);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 prod = __half22float2(__hmul2(w[j], x[j]));
            partial_sum += prod.x + prod.y;
        }

        sum += partial_sum * scale;
    }

    sum = block_reduce_sum<block_size, false>(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = __float2half(sum);
    }
}

template <int group_size>
void gemv_w8_cuda(const __half *input, const uint8_t *weight, const __half *scales, const __half *bias, __half *output,
                  int M, int N) {
    constexpr int block_size = 256;
    const int grid_size = M;
    gemv_w8_kernel<group_size, block_size><<<grid_size, block_size>>>(input, weight, scales, bias, output, N);
}

template void gemv_w8_cuda<128>(const __half *__restrict__ input, const uint8_t *__restrict__ weight,
                                const __half *__restrict__ scales, const __half *__restrict__ bias,
                                __half *__restrict__ output, int M, int N);
