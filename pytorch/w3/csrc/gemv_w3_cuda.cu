#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>

template <int warp_size = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, mask, warpSize);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    v = warp_reduce_sum(v);
    if (blockDim.x > warpSize) {
        __shared__ float shm[32];
        const int num_warps = blockDim.x / warpSize;
        const int warp_id = threadIdx.x / warpSize;
        const int lane_id = threadIdx.x % warpSize;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_sum((lane_id < num_warps) ? shm[lane_id] : 0.f);
    }
    return v;
}

template <int block_size>
__device__ __forceinline__ float block_reduce_sum_0(float v) {
    constexpr int warp_size = 32;
    v = warp_reduce_sum(v);
    if constexpr (block_size > warp_size) {
        static_assert(block_size % warp_size == 0, "invalid block size");
        constexpr int num_warps = block_size / warp_size;
        __shared__ float shm[num_warps];
        const int warp_id = threadIdx.x / warp_size;
        const int lane_id = threadIdx.x % warp_size;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_sum<num_warps>((lane_id < num_warps) ? shm[lane_id] : 0.f);
    }
    return v;
}

// template <typename scalar_t, int group_size, int block_size>
// __global__ void gemv_nf4_kernel(const scalar_t *__restrict__ input, const uint8_t *__restrict__ weight,
//                                 const float *__restrict__ scales, const scalar_t *__restrict__ bias,
//                                 scalar_t *__restrict__ output, int N) {
//     static_assert(std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, nv_bfloat16>);
//     // static_assert(sizeof(scalar_t) == 2);

//     // constexpr float quant_map[16]  {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
//     // -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
//     // 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};

//     __shared__ scalar_t scales_row[64]; // size: N / group_size

//     const uint8_t *weight_row = weight + blockDim.x * (N / 2);

//     for (int i = threadIdx.x; i < N / group_size; i += blockDim.x) {
//         scales_row[i] = scales[blockIdx.x * (N / group_size) + i];
//     }
//     __syncthreads();
//     // const float *scales_row = scales + row * (N / group_size);

//     // TODO: scales load to shm?

//     float2 sum2{0.f, 0.f};

//     for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
//         scalar_t s = scales_row[i / group_size]; // optimize?
//         __half2 s2 = make_half2(s, s);

//         uint8_t w_i[4];
//         *(int *)w_i = *(int *)&weight_row[i / 2];

//         scalar_t x_h[8];
//         *(float4 *)x_h = *(float4 *)&input[i];

//         __half2 w_f[4];

// #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             w_f[j] = make_half2(w_i[j] >> 4, w_i[j] & 4) * s2;
//         }

// #pragma unroll
//         for (int j = 0; j < 4; j++) {
//             float2 f2 = __half22float2(w_f[j]);
//             sum2.x += f2.x;
//             sum2.y += f2.y;
//         }
//     }

//     float sum = block_reduce_sum<block_size>(sum2.x + sum2.y);

//     // for (int i = 0; i < N; i ++) {
//     //     scalar_t x = input[i];
//     //     uint8_t w = weight_row[i / 2];
//     //     if (i % 2 == 0) {
//     //         w = w >> 4;
//     //     } else {
//     //         w = w & 0xf;
//     //     }
//     //     scalar_t s = scales_row[i / group_size];
//     //     scalar_t w_f = quant_map[w] * s;
//     //     scalar_t b = bias[i];
//     //     sum += x * w_f;
//     // }

//     if (threadIdx.x == 0) {
//         output[blockIdx.x] = (scalar_t)sum + bias[blockIdx.x];
//     }
// }

// template <typename scalar_t, int group_size>
// void gemv_nf4_cuda(const scalar_t *input, const uint8_t *weight, const float *scales, const scalar_t *bias,
//                    scalar_t *output, int M, int N) {
//     // const int block_size = std::max(1, N / 8);
//     constexpr int block_size = 64;
//     const int grid_size = M;
//     gemv_nf4_kernel<scalar_t, group_size, block_size>
//         <<<grid_size, block_size>>>(input, weight, scales, bias, output, N);
// }

// // template void gemv_nf4_cuda<float, 64>(const float *input, const uint8_t *weight, const float *scales, const float
// // *bias, float *output, int M, int N);

// template void gemv_nf4_cuda<half, 64>(const half *input, const uint8_t *weight, const float *scales, const half *bias,
//                                       half *output, int M, int N);

// template void gemv_nf4_cuda<nv_bfloat16, 64>(const nv_bfloat16 *input, const uint8_t *weight, const float *scales,
// const nv_bfloat16 *bias, nv_bfloat16 *output, int M, int N);

__device__ __forceinline__ __half fast_int16_to_half(int16_t x) {
    const __half hmagic = __short_as_half((0x19 << 10) + (1 << 9));
    const int16_t imagic = __half_as_short(hmagic);
    return __short_as_half(imagic + x) - hmagic;
}

template <typename scalar_t, int group_size, int block_size>
__global__ void gemv_i3_kernel(const scalar_t *__restrict__ input, const uint8_t *__restrict__ weight,
                               const scalar_t *__restrict__ scales, const scalar_t *__restrict__ bias,
                               scalar_t *__restrict__ output, int M, int N) {
                                // TODO: pass as args
    const uint8_t *weight_lo = weight + blockIdx.x * (N / 4);  // 2-bit low
    const uint8_t *weight_hi = weight + M * N / 4 + blockIdx.x * (N / 8); // 1-bit high

    __shared__ scalar_t s_scales_row[32]; // size: N / group_size = 4096 / 128
    __shared__ uint8_t s_weight_lo[1024];   // size: N / 4 = 1024
    __shared__ uint8_t s_weight_hi[512];    // size: N / 8 = 512
    // const uint8_t *weight_row = weight + blockDim.x * (N / 2);

    const scalar_t *scales_row = scales + blockIdx.x * (N / group_size);
    if (threadIdx.x < 32) {
        // N / group_size
        for (int i = threadIdx.x; i < N / group_size; i += blockDim.x) {
            s_scales_row[i] = scales_row[i];
        } 
    } else if (threadIdx.x < 32 + 64) {
        *(float4*)&s_weight_lo[(threadIdx.x - 32) * 16] = *(float4*)&weight_lo[(threadIdx.x - 32) * 16];
    } else if (threadIdx.x < 32 + 64 + 32) {
        *(float4*)&s_weight_hi[(threadIdx.x - (32+64)) * 16] = *(float4*)&weight_hi[(threadIdx.x - (32+64)) * 16];
    }
    __syncthreads();

    float sum = 0.f;

    for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
        scalar_t s = s_scales_row[i / group_size];
        // __half2 s2 = make_half2(s, s);    // TODO: bf16

        scalar_t x_h[8];
        *(float4 *)x_h = *(float4 *)&input[i];

        uint8_t w_lo_i[2];
        *(uint16_t *)w_lo_i = *(uint16_t*)&s_weight_lo[i / 4];    // TODO: optimize
        uint8_t w_hi_i = *(uint8_t*)&s_weight_hi[i / 8];      // TODO: optimize

        scalar_t w_h[8];

        // uint32_t w_i4_lo = (w_lo_i[0] | (w_lo_i[0] << (8-2)) | (w_lo_i[0] << (16-4)) | (w_lo_i[0] << (24-6))) & 0x03030303;
        // uint32_t w_i4_hi = ((w_hi_i << 2) | (w_hi_i << (8 + 1)) | (w_hi_i << 16) | (w_hi_i << (24 - 1))) & 0x04040404;
        // uint32_t w_i4 = __vsub4(w_i4_lo, w_i4_hi);

        // w_h[0] = scalar_t(int8_t(w_i4 & 0xff));
        // w_h[1] = scalar_t(int8_t((w_i4 >> 8) & 0xff));
        // w_h[2] = scalar_t(int8_t((w_i4 >> 16) & 0xff));
        // w_h[3] = scalar_t(int8_t((w_i4 >> 24) & 0xff));

        // w_i4_lo = (w_lo_i[1] | (w_lo_i[1] << (8-2)) | (w_lo_i[1] << (16-4)) | (w_lo_i[1] << (24-6))) & 0x03030303;
        // w_i4_hi = ((w_hi_i >> 2) | (w_hi_i << (8 - 3)) | (w_hi_i << (16- 4)) | (w_hi_i << (24 - 5))) & 0x04040404;
        // w_i4 = __vsub4(w_i4_lo, w_i4_hi);
        // w_h[4] = scalar_t(int8_t(w_i4 & 0xff));
        // w_h[5] = scalar_t(int8_t((w_i4 >> 8) & 0xff));
        // w_h[6] = scalar_t(int8_t((w_i4 >> 16) & 0xff));
        // w_h[7] = scalar_t(int8_t((w_i4 >> 24) & 0xff));



        w_h[0] = fast_int16_to_half(int8_t(uint32_t(w_lo_i[0]) & 0x3) - ((w_hi_i << 2) & 0x4));
        w_h[1] = fast_int16_to_half(int8_t((w_lo_i[0] >> 2) & 0x3) - ((w_hi_i << 1) & 0x4));
        w_h[2] = fast_int16_to_half(int8_t((w_lo_i[0] >> 4) & 0x3) - (w_hi_i  & 0x4));
        w_h[3] = fast_int16_to_half(int8_t(w_lo_i[0] >> 6) - ((w_hi_i >> 1) & 0x4));
        w_h[4] = fast_int16_to_half(int8_t(w_lo_i[1] & 0x3) - ((w_hi_i >> 2) & 0x4));
        w_h[5] = fast_int16_to_half(int8_t((w_lo_i[1] >> 2) & 0x3) - ((w_hi_i >> 3) & 0x4));
        w_h[6] = fast_int16_to_half(int8_t((w_lo_i[1] >> 4) & 0x3) - ((w_hi_i >> 4) & 0x4));
        w_h[7] = fast_int16_to_half(int8_t(w_lo_i[1] >> 6) - ((w_hi_i >> 5) & 0x4));

        // if (threadIdx.x + blockIdx.x + i == 0) {
        //     printf("first qweight: %f %f %f %f %f %f %f %f\n", (float) w_h[0], (float) w_h[1], (float) w_h[2], (float) w_h[3], (float) w_h[4], (float) w_h[5], (float) w_h[6],  (float) w_h[7]);
        //     printf("first weight: %f %f %f %f %f %f %f %f\n", float(s)*(float) w_h[0], float(s)*(float) w_h[1], float(s)*(float) w_h[2], float(s)*(float) w_h[3], float(s)*(float) w_h[4], float(s)*(float) w_h[5], float(s)*(float) w_h[6], float(s)* (float) w_h[7]);
        //     printf("first scale: %f\n", float(s));
        // }

        float partial_sum = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 x_w_f2 = __half22float2(__hmul2(((half2*)x_h)[j], ((half2*)w_h)[j]));
            partial_sum += x_w_f2.x + x_w_f2.y;
        }

        sum += partial_sum * float(s);
    }

    sum = block_reduce_sum_0<block_size>(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = scalar_t(sum) + bias[blockIdx.x];
    }
}

// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}


// Instances of `Vec` are used to organize groups of >>registers<<, as needed for instance as inputs to tensor core
// operations. Consequently, all corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is documented here: 
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>; // quantization scales

// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

template <typename scalar_t, int group_size, int block_size>
__global__ void gemv_i3_v2_kernel(const scalar_t *__restrict__ input, const uint8_t *__restrict__ weight,
                               const scalar_t *__restrict__ scales, const scalar_t *__restrict__ bias,
                               scalar_t *__restrict__ output) {
                                // TODO: pass as args
                                constexpr int N = 4096;
                               constexpr int chunk_size = N / 8;
    const uint8_t *weight_row0 = weight + blockIdx.x * (chunk_size * 3);
    const uint8_t* weight_row1 = weight_row0 + chunk_size;
    const uint8_t* weight_row2 = weight_row1 + chunk_size;

    const scalar_t *scales_row = scales + blockIdx.x * (N / group_size);

    float sum = 0.f;

    #pragma unroll
    for (int s =  0; s < N; s += block_size * 128) {
        const int i = s+ threadIdx.x * 128;
        float partial_sum = 0.f;

        uint w0[4];
        *(uint4*) w0 = ((uint4*)weight_row0)[i / 128];
        uint w1[4];
        *(uint4*) w1 = ((uint4*)weight_row1)[i / 128];

        half2 fb[16];
        half2 x[16];
        ((float4*)x)[0] = ((float4*)input)[i / 128 + 0 * block_size];

        ((FragB*)fb)[0] = dequant(w0[0]);
        ((FragB*)fb)[1] = dequant(w0[0] >> 4);

        ((float4*)x)[1] = ((float4*)input)[i / 128 + 1 * block_size];

        ((FragB*)fb)[2] = dequant(w0[1]);
        ((FragB*)fb)[3] = dequant(w0[1] >> 4);

        ((float4*)x)[2] = ((float4*)input)[i / 128 + 2 * block_size];

        ((FragB*)fb)[4] = dequant(w0[2]);
        ((FragB*)fb)[5] = dequant(w0[2] >> 4);

        ((float4*)x)[3] = ((float4*)input)[i / 128 + 3 * block_size];

        ((FragB*)fb)[6] = dequant(w0[3]);
        ((FragB*)fb)[7] = dequant(w0[3] >> 4);

        uint w2[4];
        *(uint4*) w2 = ((uint4*)weight_row2)[i / 128];

        float2 f2;
        #pragma unroll
        for (int j  = 0; j < 16; j++) {
            f2 = __half22float2(__hmul2(x[j], fb[j]));
            partial_sum += f2.x + f2.y;
        }

        ((float4*)x)[0] = ((float4*)input)[i / 128 + 4 * block_size];
        ((FragB*)fb)[0] = dequant(w1[0]);
        ((FragB*)fb)[1] = dequant(w1[0] >> 4);
        ((float4*)x)[1] = ((float4*)input)[i / 128 + 5 * block_size];
        ((FragB*)fb)[2] = dequant(w1[1]);
        ((FragB*)fb)[3] = dequant(w1[1] >> 4);
        ((float4*)x)[2] = ((float4*)input)[i / 128 + 6 * block_size];
        ((FragB*)fb)[4] = dequant(w1[2]);
        ((FragB*)fb)[5] = dequant(w1[2] >> 4);
        ((float4*)x)[3] = ((float4*)input)[i / 128 + 7 * block_size];
        ((FragB*)fb)[6] = dequant(w1[3]);
        ((FragB*)fb)[7] = dequant(w1[3] >> 4);

        #pragma unroll
        for (int j  = 0; j < 16; j++) {
            f2 = __half22float2(__hmul2(x[j], fb[j]));
            partial_sum += f2.x + f2.y;
            // f2 = __half22float2(x[j]);
            // partial_sum += f2.x + f2.y;
            // f2 = __half22float2(fb[j]);
            // partial_sum += f2.x + f2.y;
        }

        ((float4*)x)[0] = ((float4*)input)[i / 128 + 8 * block_size];
        ((FragB*)fb)[0] = dequant(w2[0]);
        ((FragB*)fb)[1] = dequant(w2[0] >> 4);
        ((float4*)x)[1] = ((float4*)input)[i / 128 + 9 * block_size];
        ((FragB*)fb)[2] = dequant(w2[1]);
        ((FragB*)fb)[3] = dequant(w2[1] >> 4);
        ((float4*)x)[2] = ((float4*)input)[i / 128 + 10 * block_size];
        ((FragB*)fb)[4] = dequant(w2[2]);
        ((FragB*)fb)[5] = dequant(w2[2] >> 4);
        ((float4*)x)[3] = ((float4*)input)[i / 128 + 11 * block_size];
        ((FragB*)fb)[6] = dequant(w2[3]);
        ((FragB*)fb)[7] = dequant(w2[3] >> 4);

        #pragma unroll
        for (int j  = 0; j < 16; j++) {
            f2 = __half22float2(__hmul2(x[j], fb[j]));
            partial_sum += f2.x + f2.y;
            // f2 = __half22float2(x[j]);
            // partial_sum += f2.x + f2.y;
            // f2 = __half22float2(fb[j]);
            // partial_sum += f2.x + f2.y;
        }

        scalar_t sc = scales_row[i / group_size];

        ((float4*)x)[0] = ((float4*)input)[i / 128 + 12 * block_size];
        ((float4*)x)[1] = ((float4*)input)[i / 128 + 13 * block_size];
        ((float4*)x)[2] = ((float4*)input)[i / 128 + 14 * block_size];
        ((float4*)x)[3] = ((float4*)input)[i / 128 + 15 * block_size];

        uint32_t w4_x = (w0[0] >> 3) | (w1[0] >> 2) | (w2[0] >> 1);
        // uint32_t w4_x = (w0[0] >> 3);
        ((FragB*)fb)[0] = dequant(w4_x);
        ((FragB*)fb)[1] = dequant(w4_x >> 4);
        uint32_t w4_y = (w0[1] >> 3) | (w1[1] >> 2) | (w2[1] >> 1);
        // uint32_t w4_y = (w0[1] >> 3) ;
        ((FragB*)fb)[2] = dequant(w4_y);
        ((FragB*)fb)[3] = dequant(w4_y >> 4);
        uint32_t w4_z = (w0[2] >> 3) | (w1[2] >> 2) | (w2[2] >> 1);
        // uint32_t w4_z = (w0[2] >> 3) ;
        ((FragB*)fb)[4] = dequant(w4_z);
        ((FragB*)fb)[5] = dequant(w4_z >> 4);
        uint32_t w4_w = (w0[3] >> 3) | (w1[3] >> 2) | (w2[3] >> 1);
        // uint32_t w4_w = (w0[3] >> 3) ;
        ((FragB*)fb)[6] = dequant(w4_w);
        ((FragB*)fb)[7] = dequant(w4_w >> 4);

        #pragma unroll
        for (int j  = 0; j < 16; j++) {
            f2 = __half22float2(__hmul2(x[j], fb[j]));
            partial_sum += f2.x + f2.y;
            // f2 = __half22float2(x[j]);
            // partial_sum += f2.x + f2.y;
            // f2 = __half22float2(fb[j]);
            // partial_sum += f2.x + f2.y;
        }

        sum += partial_sum * float(sc);
    }

    sum = block_reduce_sum_0<block_size>(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = scalar_t(sum) + bias[blockIdx.x];
    }
}

void gemv_i3_cuda(const __half *input, const uint8_t *weight, const __half *scales, const __half* bias, __half *output, int M, int N) {
    constexpr int block_size = 32;
    constexpr int group_size = 128;
    const int grid_size = M;
    gemv_i3_v2_kernel<__half, group_size, block_size>
        <<<grid_size, block_size>>>(input, weight, scales, bias, output);
}
