// Tutorial: https://siboehm.com/articles/22/CUDA-MMM
// CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

#include "common.h"
#include <functional>
#include <vector>

__global__ void sgemm1_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        float s = 0.f;
        for (int k = 0; k < K; k++) {
            s += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = s;
    }
}

static inline void sgemm1(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_DIM_X = 32;
    constexpr int BLOCK_DIM_Y = 32;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM_X), ceil_div(M, BLOCK_DIM_Y));
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm1_kernel<<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

template <int BLOCK_DIM>
__global__ void sgemm2_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

    A += by * BLOCK_DIM * K;
    B += bx * BLOCK_DIM;
    C += (by * N + bx) * BLOCK_DIM;

    float s = 0;
    for (int bk = 0; bk < K; bk += BLOCK_DIM) {
        As[ty][tx] = A[ty * K + tx];
        Bs[ty][tx] = B[ty * N + tx];
        __syncthreads();
#pragma unroll
        for (int tk = 0; tk < BLOCK_DIM; tk++) {
            s += As[ty][tk] * Bs[tk][tx];
        }
        A += BLOCK_DIM;
        B += BLOCK_DIM * N;
        __syncthreads();
    }
    C[ty * N + tx] = s;
}

static inline void sgemm2(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_DIM = 32;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM), ceil_div(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    sgemm2_kernel<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

__device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 &operator+=(float4 &self, const float4 &other) { return self = self + other; }

__device__ __forceinline__ float4 operator*(const float4 &a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 &operator+=(float2 &self, const float2 &other) { return self = self + other; }

__device__ __forceinline__ float2 operator*(const float2 &a, float s) { return make_float2(a.x * s, a.y * s); }

template <size_t N>
struct FloatN {
    using type = std::conditional_t<N % 4 == 0, float4, std::conditional_t<N % 2 == 0, float2, float>>;
};

template <int BM, int BN, int BK, int TM, int TN, int TK, bool PREFETCH_GLOBAL, bool PREFETCH_SHARED, int GM>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm3_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                  float *__restrict__ C) {
    static_assert(TN == 1 || TN == 2 || TN % 4 == 0);
    static_assert(TK == 1 || TK == 2 || TK % 4 == 0);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if constexpr (GM > 1) {
        // L2 cache optimization: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
        const int bid = by * gridDim.x + bx;
        const int num_blocks_m = (M + (BM - 1)) / BM;
        const int num_blocks_n = (N + (BN - 1)) / BN;
        const int num_blocks_in_group = GM * num_blocks_n;
        const int group_id = bid / num_blocks_in_group;
        const int first_block_m = group_id * GM;
        const int gm = min(num_blocks_m - first_block_m, GM);
        const int bid_in_group = bid - group_id * num_blocks_in_group;
        bx = bid_in_group / gm;
        by = first_block_m + (bid_in_group - bx * gm);
    }

    const int tid = ty * blockDim.x + tx;

    constexpr int NUM_SHM_BUF = PREFETCH_GLOBAL ? 2 : 1;
    __shared__ float As[NUM_SHM_BUF][BM * BK];
    __shared__ float Bs[NUM_SHM_BUF][BK * BN];

    float sums[TM * TN] = {};

    A += by * BM * K;           // move A to top-left corner of first row block
    B += bx * BN;               // move B to top-left corner of first column block
    C += by * BM * N + bx * BN; // move C to the beginning of the result block

    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    // constants for loading A/B from global memory into shared memory
    static_assert((BM * BK) % (NUM_THREADS * 4) == 0);
    constexpr int A_LOAD_TILE_Y = (NUM_THREADS * 4 < BK) ? 1 : NUM_THREADS * 4 / BK;
    constexpr int A_LOAD_TILE_X = (NUM_THREADS * 4 < BK) ? NUM_THREADS * 4 : BK;
    const int A_y_offset = tid * 4 / BK;
    const int A_x_offset = tid * 4 % BK;

    static_assert((BK * BN) % (NUM_THREADS * 4) == 0);
    constexpr int B_LOAD_TILE_Y = (NUM_THREADS * 4 < BN) ? 1 : NUM_THREADS * 4 / BN;
    constexpr int B_LOAD_TILE_X = (NUM_THREADS * 4 < BN) ? NUM_THREADS * 4 : BN;
    const int B_y_offset = tid * 4 / BN;
    const int B_x_offset = tid * 4 % BN;

    using float_tk = typename FloatN<TK>::type;
    constexpr size_t float_tk_n = sizeof(float_tk) / sizeof(float);

    using float_tn = typename FloatN<TN>::type;
    constexpr size_t float_tn_n = sizeof(float_tn) / sizeof(float);

    // constants & registers for prefetch
    constexpr int PREFETCH_A_STEPS_Y = BM / A_LOAD_TILE_Y;
    constexpr int PREFETCH_A_STEPS_X = BK / A_LOAD_TILE_X;
    constexpr int PREFETCH_A_STEPS = PREFETCH_A_STEPS_Y * PREFETCH_A_STEPS_X;
    constexpr int PREFETCH_B_STEPS_Y = BK / B_LOAD_TILE_Y;
    constexpr int PREFETCH_B_STEPS_X = BN / B_LOAD_TILE_X;
    constexpr int PREFETCH_B_STEPS = PREFETCH_B_STEPS_Y * PREFETCH_B_STEPS_X;

    constexpr int BK_STEPS = BK / TK;
    constexpr int PREFETCH_A_EVERY_BK_STEPS = (BK_STEPS >= PREFETCH_A_STEPS) ? BK_STEPS / PREFETCH_A_STEPS : 1;
    constexpr int PREFETCH_A_NUM_PER_STEP = PREFETCH_A_STEPS / (BK_STEPS / PREFETCH_A_EVERY_BK_STEPS);
    constexpr int PREFETCH_B_EVERY_BK_STEPS = (BK_STEPS >= PREFETCH_B_STEPS) ? BK_STEPS / PREFETCH_B_STEPS : 1;
    constexpr int PREFETCH_B_NUM_PER_STEP = PREFETCH_B_STEPS / (BK_STEPS / PREFETCH_B_EVERY_BK_STEPS);

    int buf_idx = 0;

    for (int k = 0; k < K; k += BK) {
        if (!PREFETCH_GLOBAL || k == 0) {
            // each block loads A[0:BM][0:BK] into As
#pragma unroll
            for (int A_y_start = 0; A_y_start < BM; A_y_start += A_LOAD_TILE_Y) {
                const int A_y = A_y_start + A_y_offset;
#pragma unroll
                for (int A_x_start = 0; A_x_start < BK; A_x_start += A_LOAD_TILE_X) {
                    const int A_x = A_x_start + A_x_offset;
                    *(float4 *)&As[0][A_y * BK + A_x] = *(float4 *)&A[A_y * K + A_x];
                }
            }
            // each block loads B[0:BK][0:BN] into Bs
#pragma unroll
            for (int B_y_start = 0; B_y_start < BK; B_y_start += B_LOAD_TILE_Y) {
                const int B_y = B_y_start + B_y_offset;
#pragma unroll
                for (int B_x_start = 0; B_x_start < BN; B_x_start += B_LOAD_TILE_X) {
                    const int B_x = B_x_start + B_x_offset;
                    *(float4 *)&Bs[0][B_y * BN + B_x] = *(float4 *)&B[B_y * N + B_x];
                }
            }
        }

        const float *pAs = As[buf_idx] + ty * TM * BK;
        const float *pBs = Bs[buf_idx] + tx * TN;

        if constexpr (PREFETCH_GLOBAL) {
            buf_idx ^= 1;
        }

        const float *A_next = A + BK;
        const float *B_next = B + BK * N;

        __syncthreads();

#pragma unroll
        for (int bk = 0; bk < BK; bk += TK) {
            if (PREFETCH_GLOBAL && k + BK < K) {
                if (bk % (TK * PREFETCH_A_EVERY_BK_STEPS) == 0) {
                    // prefetch next float4 of As tile from global memory into register
                    const int A_prefetch_idx_start = bk / (TK * PREFETCH_A_EVERY_BK_STEPS) * PREFETCH_A_NUM_PER_STEP;
#pragma unroll
                    for (int A_prefetch_idx_offset = 0; A_prefetch_idx_offset < PREFETCH_A_NUM_PER_STEP;
                         A_prefetch_idx_offset++) {
                        const int A_prefetch_idx = A_prefetch_idx_start + A_prefetch_idx_offset;
                        const int A_prefetch_y = A_prefetch_idx / PREFETCH_A_STEPS_X;
                        const int A_prefetch_x = A_prefetch_idx % PREFETCH_A_STEPS_X;
                        const int A_y = A_prefetch_y * A_LOAD_TILE_Y + A_y_offset;
                        const int A_x = A_prefetch_x * A_LOAD_TILE_X + A_x_offset;
                        *(float4 *)&As[buf_idx][A_y * BK + A_x] = *(float4 *)&A_next[A_y * K + A_x];
                    }
                }
                if ((bk + TK) % (TK * PREFETCH_B_EVERY_BK_STEPS) == 0) {
                    // prefetch next float4 of Bs tile from global memory into register (interleaved with As)
                    const int B_prefetch_idx_start = bk / (TK * PREFETCH_B_EVERY_BK_STEPS) * PREFETCH_B_NUM_PER_STEP;
#pragma unroll
                    for (int B_prefetch_idx_offset = 0; B_prefetch_idx_offset < PREFETCH_B_NUM_PER_STEP;
                         B_prefetch_idx_offset++) {
                        const int B_prefetch_idx = B_prefetch_idx_start + B_prefetch_idx_offset;
                        const int B_prefetch_y = B_prefetch_idx / PREFETCH_B_STEPS_X;
                        const int B_prefetch_x = B_prefetch_idx % PREFETCH_B_STEPS_X;
                        const int B_y = B_prefetch_y * B_LOAD_TILE_Y + B_y_offset;
                        const int B_x = B_prefetch_x * B_LOAD_TILE_X + B_x_offset;
                        *(float4 *)&Bs[buf_idx][B_y * BN + B_x] = *(float4 *)&B_next[B_y * N + B_x];
                    }
                }
            }
            constexpr int NUM_REG_BUF = PREFETCH_SHARED ? 2 : 1;
            // minimize register usage
            if constexpr (TM <= TN) {
                float Areg[TM * TK];
                // each thread loads pAs[0:TM][0:TK] into Areg
#pragma unroll
                for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                    for (int tk = 0; tk < TK; tk += float_tk_n) {
                        *(float_tk *)&Areg[tm * TK + tk] = *(float_tk *)&pAs[tm * BK + tk];
                    }
                }
                // each thread loads pBs[0:TK][0:TN] into Breg and compute matmul
                float_tn Breg[NUM_REG_BUF];
                int reg_buf_idx = 0;
#pragma unroll
                for (int tk = 0; tk < TK; tk++) {
#pragma unroll
                    for (int tn = 0; tn < TN; tn += float_tn_n) {
                        if (!PREFETCH_SHARED || tk + tn == 0) {
                            Breg[0] = *(float_tn *)&pBs[tk * BN + tn];
                        }
                        if constexpr (PREFETCH_SHARED) {
                            if (tn + float_tn_n < TN) {
                                Breg[reg_buf_idx ^ 1] = *(float_tn *)&pBs[tk * BN + tn + float_tn_n];
                            } else if (tk + 1 < TK) {
                                Breg[reg_buf_idx ^ 1] = *(float_tn *)&pBs[(tk + 1) * BN];
                            }
                        }
#pragma unroll
                        for (int tm = 0; tm < TM; tm++) {
                            *(float_tn *)&sums[tm * TN + tn] += Breg[reg_buf_idx] * Areg[tm * TK + tk];
                        }
                        if constexpr (PREFETCH_SHARED) {
                            reg_buf_idx ^= 1;
                        }
                    }
                }
            } else {
                float Breg[TK * TN];
                // each thread loads pBs[0:TK][0:TN] into Breg
#pragma unroll
                for (int tk = 0; tk < TK; tk++) {
#pragma unroll
                    for (int tn = 0; tn < TN; tn += float_tn_n) {
                        *(float_tn *)&Breg[tk * TN + tn] = *(float_tn *)&pBs[tk * BN + tn];
                    }
                }
                // each thread loads pAs[0:TM][0:TK] into Areg and compute matmul
                float_tk Areg[NUM_REG_BUF];
                int reg_buf_idx = 0;
#pragma unroll
                for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                    for (int tk = 0; tk < TK; tk += float_tk_n) {
                        if (!PREFETCH_SHARED || tm + tk == 0) {
                            Areg[0] = *(float_tk *)&pAs[tm * BK + tk];
                        }
                        if constexpr (PREFETCH_SHARED) {
                            if (tk + float_tk_n < TK) {
                                Areg[reg_buf_idx ^ 1] = *(float_tk *)&pAs[tm * BK + tk + float_tk_n];
                            } else if (tm + 1 < TM) {
                                Areg[reg_buf_idx ^ 1] = *(float_tk *)&pAs[(tm + 1) * BK];
                            }
                        }
#pragma unroll
                        for (int tsubk = 0; tsubk < float_tk_n; tsubk++) {
#pragma unroll
                            for (int tn = 0; tn < TN; tn++) {
                                sums[tm * TN + tn] +=
                                    Breg[(tk + tsubk) * TN + tn] * ((float *)&Areg[reg_buf_idx])[tsubk];
                            }
                        }
                        if constexpr (PREFETCH_SHARED) {
                            reg_buf_idx ^= 1;
                        }
                    }
                }
            }

            pAs += TK;
            pBs += TK * BN;
        }

        A = A_next;
        B = B_next;

        if constexpr (!PREFETCH_GLOBAL) {
            __syncthreads();
        }
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn += float_tn_n) {
            *(float_tn *)&C[(ty * TM + tm) * N + tx * TN + tn] = *(float_tn *)&sums[tm * TN + tn];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int TM = 4, int TN = 4, int TK = 4, bool PREFETCH_GLOBAL = true,
          bool PREFETCH_SHARED = false, int GM = 1>
static inline void sgemm3(int M, int N, int K, const float *A, const float *B, float *C) {
    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "invalid matrix dimensions";

    static_assert(BM % TM == 0 && BN % TN == 0 && BK % TK == 0);

    constexpr int BLOCK_DIM_X = BN / TN;
    constexpr int BLOCK_DIM_Y = BM / TM;
    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024);

    dim3 grid_dim(N / BN, M / BM);
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm3_kernel<BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, GM>
        <<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int TK, bool PREFETCH_GLOBAL>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * WARP_SIZE)
    sgemm4_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                  float *__restrict__ C) {
    static_assert(TN % 4 == 0); // float4
    static_assert(TK == 1 || TK == 2 || TK % 4 == 0);

    const int tid = threadIdx.x; // thread id

    // warp tiling
    constexpr int NUM_WARPS_M = BM / WM;
    constexpr int NUM_WARPS_N = BN / WN;
    static_assert(WN % WNITER == 0);
    constexpr int WSUBN = WN / WNITER;
    static_assert(WSUBN % TN == 0);
    constexpr int NUM_LANES_N = WSUBN / TN;
    static_assert(WARP_SIZE % NUM_LANES_N == 0);
    constexpr int NUM_LANES_M = WARP_SIZE / NUM_LANES_N;
    constexpr int WSUBM = NUM_LANES_M * TM;
    static_assert(WM % WSUBM == 0);
    constexpr int WMITER = WM / WSUBM;

    const int warp_id = tid / WARP_SIZE;
    const int wm = warp_id / NUM_WARPS_N; // warp row index
    const int wn = warp_id % NUM_WARPS_N; // warp column index

    const int lane_id = tid % WARP_SIZE;
    const int lm = lane_id / NUM_LANES_N; // lane row index
    const int ln = lane_id % NUM_LANES_N; // lane column index

    constexpr int NUM_THREADS = NUM_WARPS_M * NUM_WARPS_N * WARP_SIZE;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float Areg[WMITER * TM * TK];
    float sums[WMITER * WNITER * TM * TN] = {};

    A += by * BM * K; // move A to top-left corner of first row block
    B += bx * BN;     // move B to top-left corner of first column block

    // constants for loading A/B from global memory into shared memory
    static_assert((BM * BK) % (NUM_THREADS * 4) == 0);
    constexpr int A_LOAD_TILE_Y = (NUM_THREADS * 4 < BK) ? 1 : NUM_THREADS * 4 / BK;
    constexpr int A_LOAD_TILE_X = (NUM_THREADS * 4 < BK) ? NUM_THREADS * 4 : BK;
    const int A_y_offset = tid * 4 / BK;
    const int A_x_offset = tid * 4 % BK;

    static_assert((BK * BN) % (NUM_THREADS * 4) == 0);
    constexpr int B_LOAD_TILE_Y = (NUM_THREADS * 4 < BN) ? 1 : NUM_THREADS * 4 / BN;
    constexpr int B_LOAD_TILE_X = (NUM_THREADS * 4 < BN) ? NUM_THREADS * 4 : BN;
    const int B_y_offset = tid * 4 / BN;
    const int B_x_offset = tid * 4 % BN;

    // constants & registers for prefetch
    constexpr int PREFETCH_A_STEPS_Y = BM / A_LOAD_TILE_Y;
    constexpr int PREFETCH_A_STEPS_X = BK / A_LOAD_TILE_X;
    constexpr int PREFETCH_A_STEPS = PREFETCH_A_STEPS_Y * PREFETCH_A_STEPS_X;
    float4 As_next[PREFETCH_A_STEPS];
    constexpr int PREFETCH_B_STEPS_Y = BK / B_LOAD_TILE_Y;
    constexpr int PREFETCH_B_STEPS_X = BN / B_LOAD_TILE_X;
    constexpr int PREFETCH_B_STEPS = PREFETCH_B_STEPS_Y * PREFETCH_B_STEPS_X;
    float4 Bs_next[PREFETCH_B_STEPS];

    constexpr int BK_STEPS = BK / TK;
    constexpr int PREFETCH_A_EVERY_BK_STEPS = (BK_STEPS >= PREFETCH_A_STEPS) ? BK_STEPS / PREFETCH_A_STEPS : 1;
    constexpr int PREFETCH_A_NUM_PER_STEP = PREFETCH_A_STEPS / (BK_STEPS / PREFETCH_A_EVERY_BK_STEPS);
    constexpr int PREFETCH_B_EVERY_BK_STEPS = (BK_STEPS >= PREFETCH_B_STEPS) ? BK_STEPS / PREFETCH_B_STEPS : 1;
    constexpr int PREFETCH_B_NUM_PER_STEP = PREFETCH_B_STEPS / (BK_STEPS / PREFETCH_B_EVERY_BK_STEPS);

    for (int k = 0; k < K; k += BK) {
        if (!PREFETCH_GLOBAL || k == 0) {
            // each block loads A[0:BM][0:BK] into As
#pragma unroll
            for (int A_y_start = 0; A_y_start < BM; A_y_start += A_LOAD_TILE_Y) {
                const int A_y = A_y_start + A_y_offset;
#pragma unroll
                for (int A_x_start = 0; A_x_start < BK; A_x_start += A_LOAD_TILE_X) {
                    const int A_x = A_x_start + A_x_offset;
                    *(float4 *)&As[A_y * BK + A_x] = *(float4 *)&A[A_y * K + A_x];
                }
            }
            // each block loads B[0:BK][0:BN] into Bs
#pragma unroll
            for (int B_y_start = 0; B_y_start < BK; B_y_start += B_LOAD_TILE_Y) {
                const int B_y = B_y_start + B_y_offset;
#pragma unroll
                for (int B_x_start = 0; B_x_start < BN; B_x_start += B_LOAD_TILE_X) {
                    const int B_x = B_x_start + B_x_offset;
                    *(float4 *)&Bs[B_y * BN + B_x] = *(float4 *)&B[B_y * N + B_x];
                }
            }
        } else {
            // load prefetched next tile into As
#pragma unroll
            for (int A_y_start = 0; A_y_start < BM; A_y_start += A_LOAD_TILE_Y) {
                const int A_y = A_y_start + A_y_offset;
                const int A_next_y = A_y_start / A_LOAD_TILE_Y;
#pragma unroll
                for (int A_x_start = 0; A_x_start < BK; A_x_start += A_LOAD_TILE_X) {
                    const int A_x = A_x_start + A_x_offset;
                    const int A_next_x = A_x_start / A_LOAD_TILE_X;
                    *(float4 *)&As[A_y * BK + A_x] = As_next[A_next_y * PREFETCH_A_STEPS_X + A_next_x];
                }
            }
            // load prefetched next tile into Bs
#pragma unroll
            for (int B_y_start = 0; B_y_start < BK; B_y_start += B_LOAD_TILE_Y) {
                const int B_y = B_y_start + B_y_offset;
                const int B_next_y = B_y_start / B_LOAD_TILE_Y;
#pragma unroll
                for (int B_x_start = 0; B_x_start < BN; B_x_start += B_LOAD_TILE_X) {
                    const int B_x = B_x_start + B_x_offset;
                    const int B_next_x = B_x_start / B_LOAD_TILE_X;
                    *(float4 *)&Bs[B_y * BN + B_x] = Bs_next[B_next_y * PREFETCH_B_STEPS_X + B_next_x];
                }
            }
        }
        __syncthreads();

        const float *pAs = As + wm * WM * BK;
        const float *pBs = Bs + wn * WN;

        const float *A_next = A + BK;
        const float *B_next = B + BK * N;

#pragma unroll
        for (int bk = 0; bk < BK; bk += TK) {
            if constexpr (PREFETCH_GLOBAL) {
                if (k + BK < K && bk % (TK * PREFETCH_A_EVERY_BK_STEPS) == 0) {
                    // prefetch next float4 of As tile from global memory into register
                    const int A_prefetch_idx_start = bk / (TK * PREFETCH_A_EVERY_BK_STEPS) * PREFETCH_A_NUM_PER_STEP;
#pragma unroll
                    for (int A_prefetch_idx_offset = 0; A_prefetch_idx_offset < PREFETCH_A_NUM_PER_STEP;
                         A_prefetch_idx_offset++) {
                        const int A_prefetch_idx = A_prefetch_idx_start + A_prefetch_idx_offset;
                        const int A_prefetch_y = A_prefetch_idx / PREFETCH_A_STEPS_X;
                        const int A_prefetch_x = A_prefetch_idx % PREFETCH_A_STEPS_X;
                        const int A_y = A_prefetch_y * A_LOAD_TILE_Y + A_y_offset;
                        const int A_x = A_prefetch_x * A_LOAD_TILE_X + A_x_offset;
                        As_next[A_prefetch_idx] = *(float4 *)&A_next[A_y * K + A_x];
                    }
                }
                if (k + BK < K && bk % (TK * PREFETCH_B_EVERY_BK_STEPS) == 0) {
                    // prefetch next float4 of Bs tile from global memory into register
                    const int B_prefetch_idx_start = bk / (TK * PREFETCH_B_EVERY_BK_STEPS) * PREFETCH_B_NUM_PER_STEP;
#pragma unroll
                    for (int B_prefetch_idx_offset = 0; B_prefetch_idx_offset < PREFETCH_B_NUM_PER_STEP;
                         B_prefetch_idx_offset++) {
                        const int B_prefetch_idx = B_prefetch_idx_start + B_prefetch_idx_offset;
                        const int B_prefetch_y = B_prefetch_idx / PREFETCH_B_STEPS_X;
                        const int B_prefetch_x = B_prefetch_idx % PREFETCH_B_STEPS_X;
                        const int B_y = B_prefetch_y * B_LOAD_TILE_Y + B_y_offset;
                        const int B_x = B_prefetch_x * B_LOAD_TILE_X + B_x_offset;
                        Bs_next[B_prefetch_idx] = *(float4 *)&B_next[B_y * N + B_x];
                    }
                }
            }
            // each warp loads the entire warp tile into registers
            const float *tAs = pAs + lm * TM * BK;
            const float *tAreg = Areg;

#pragma unroll
            for (int wmiter = 0; wmiter < WMITER; wmiter++) {
                // each thread loads tAs[0:TM][0:TK] into tAreg
#pragma unroll
                for (int tm = 0; tm < TM; tm++) {
                    using float_tk = typename FloatN<TK>::type;
                    constexpr size_t float_tk_n = sizeof(float_tk) / sizeof(float);
#pragma unroll
                    for (int tk = 0; tk < TK; tk += float_tk_n) {
                        *(float_tk *)&tAreg[tm * TK + tk] = *(float_tk *)&tAs[tm * BK + tk];
                    }
                }
                tAs += WSUBM * BK;
                tAreg += TM * TK;
            }

            const float *tBs = pBs + ln * TN;
#pragma unroll
            for (int wniter = 0; wniter < WNITER; wniter++) {
                // each thread loads tBs[0:TK][0:TN] into Breg and compute matmul
#pragma unroll
                for (int tk = 0; tk < TK; tk++) {
#pragma unroll
                    for (int tn = 0; tn < TN; tn += 4) {
                        float4 Breg = *(float4 *)&tBs[tk * BN + tn];
#pragma unroll
                        for (int wmiter = 0; wmiter < WMITER; wmiter++) {
#pragma unroll
                            for (int tm = 0; tm < TM; tm++) {
                                *(float4 *)&sums[(wmiter * WNITER + wniter) * TM * TN + tm * TN + tn] +=
                                    Breg * Areg[wmiter * TM * TK + tm * TK + tk];
                            }
                        }
                    }
                }
                tBs += WSUBN;
            }

            pAs += TK;
            pBs += TK * BN;
        }

        A += BK;
        B += BK * N;

        __syncthreads();
    }

    // move C to the top-left corner of warp tile, so that each warp takes care of C[0:WM][0:WN]
    C += (by * BM + wm * WM) * N + bx * BN + wn * WN;
#pragma unroll
    for (int wmiter = 0; wmiter < WMITER; wmiter++) {
#pragma unroll
        for (int wniter = 0; wniter < WNITER; wniter++) {
            float *tC = C + (wmiter * WSUBM + lm * TM) * N + wniter * WSUBN + ln * TN;
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn += 4) {
                    *(float4 *)&tC[tm * N + tn] = *(float4 *)&sums[(wmiter * WNITER + wniter) * TM * TN + tm * TN + tn];
                }
            }
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int WM = 16, int WN = 32, int WNITER = 1, int TM = 4, int TN = 4,
          int TK = 4, bool PREFETCH_GLOBAL = true>
static inline void sgemm4(int M, int N, int K, const float *A, const float *B, float *C) {
    static_assert(BM % WM == 0 && BN % WN == 0);
    constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARP_SIZE;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024);

    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "invalid matrix dimensions";
    dim3 grid_dim(N / BN, M / BM);
    dim3 block_dim(NUM_THREADS);
    sgemm4_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, TK, PREFETCH_GLOBAL><<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

static inline void cublas_sgemm(cublasHandle_t handle, int M, int N, int K, const float *dA, const float *dB,
                                float *dC) {
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
}

void perf(int M, int N, int K) {
    // make data
    float *A = (float *)malloc(sizeof(float) * M * K);
    for (int i = 0; i < M * K; i++) {
        A[i] = uniform();
    }

    float *B = (float *)malloc(sizeof(float) * K * N);
    for (int i = 0; i < K * N; i++) {
        B[i] = uniform();
    }

    float *dA;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));

    float *dB;
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::vector<std::pair<std::string, std::function<void(int, int, int, const float *, const float *, float *)>>>
        kernels{
            {"sgemm1", sgemm1},
            {"sgemm2", sgemm2},
            {"cublas", [handle](int M, int N, int K, const float *dA, const float *dB,
                                float *dC) { cublas_sgemm(handle, M, N, K, dA, dB, dC); }},
        };

#define TRY_DIV(a, b) ((b != 0) ? (a) / (b) : 0)

#define ADD_KERNEL(stmt) kernels.emplace_back(#stmt, (stmt))

#define ADD_SGEMM3(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, GM)                                       \
    do {                                                                                                               \
        constexpr int NUM_THREADS = (BM / TM) * (BN / TN);                                                             \
        constexpr bool is_valid_num_threads = (32 <= NUM_THREADS) && (NUM_THREADS <= 1024);                            \
        constexpr bool is_shm_oom = (PREFETCH_GLOBAL ? 2 : 1) * (BM * BK + BK * BN) * sizeof(float) > 0xc000;          \
        constexpr bool is_valid_load_tile =                                                                            \
            ((BK * BN) % (NUM_THREADS * 4) == 0) && ((BM * BK) % (NUM_THREADS * 4) == 0);                              \
        if constexpr (is_valid_num_threads && !is_shm_oom && is_valid_load_tile) {                                     \
            ADD_KERNEL((sgemm3<BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, GM>));                        \
        }                                                                                                              \
    } while (0)

#define ADD_SGEMM3_GM(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED)                                        \
    ADD_SGEMM3(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, 1)

    // ;                                           \
    // ADD_SGEMM3(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, 2);                                           \
    // ADD_SGEMM3(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, 4);                                           \
    // ADD_SGEMM3(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, PREFETCH_SHARED, 8)

#define ADD_SGEMM3_PREFETCH_SHARED(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL)                                            \
    ADD_SGEMM3_GM(BM, BN, BK, TM, TN, TK, PREFETCH_GLOBAL, true)

#define ADD_SGEMM3_PREFETCH_GLOBAL(BM, BN, BK, TM, TN, TK) ADD_SGEMM3_PREFETCH_SHARED(BM, BN, BK, TM, TN, TK, true)

#define ADD_SGEMM3_TK(BM, BN, BK, TM, TN)                                                                              \
    ADD_SGEMM3_PREFETCH_GLOBAL(BM, BN, BK, TM, TN, 1);                                                                 \
    ADD_SGEMM3_PREFETCH_GLOBAL(BM, BN, BK, TM, TN, 2);                                                                 \
    ADD_SGEMM3_PREFETCH_GLOBAL(BM, BN, BK, TM, TN, 4)

#define ADD_SGEMM3_TN(BM, BN, BK, TM)                                                                                  \
    ADD_SGEMM3_TK(BM, BN, BK, TM, 1);                                                                                  \
    ADD_SGEMM3_TK(BM, BN, BK, TM, 2);                                                                                  \
    ADD_SGEMM3_TK(BM, BN, BK, TM, 4)

#define ADD_SGEMM3_TM(BM, BN, BK)                                                                                      \
    ADD_SGEMM3_TN(BM, BN, BK, 1);                                                                                      \
    ADD_SGEMM3_TN(BM, BN, BK, 2);                                                                                      \
    ADD_SGEMM3_TN(BM, BN, BK, 4);                                                                                      \
    ADD_SGEMM3_TN(BM, BN, BK, 8)

#define ADD_SGEMM3_BK(BM, BN)                                                                                          \
    ADD_SGEMM3_TM(BM, BN, 32);                                                                                         \
    ADD_SGEMM3_TM(BM, BN, 64);                                                                                         \
    ADD_SGEMM3_TM(BM, BN, 128)

#define ADD_SGEMM3_BN(BM)                                                                                              \
    ADD_SGEMM3_BK(BM, 32);                                                                                             \
    ADD_SGEMM3_BK(BM, 64);                                                                                             \
    ADD_SGEMM3_BK(BM, 128)

#define ADD_SGEMM3_ALL                                                                                                 \
    ADD_SGEMM3_BN(32);                                                                                                 \
    ADD_SGEMM3_BN(64);                                                                                                 \
    ADD_SGEMM3_BN(128)

    ADD_SGEMM3_ALL;

    // best kernels on V100-SXM2
    // ADD_SGEMM3(32, 32, 32, 2, 4, 1, true, false, 1);  // best for 128 & 256
    // ADD_SGEMM3(64, 64, 32, 4, 4, 1, true, false, 1);  // best for 512
    // ADD_SGEMM3(32, 64, 32, 4, 4, 2, false, false, 8); // best for 1024
    // ADD_SGEMM3(64, 64, 64, 4, 4, 8, true, false, 8);  // best for 2048
    // ADD_SGEMM3(128, 64, 32, 8, 4, 1, true, false, 4); // best for 4096

    // // best kernels on A100-SXM4-80GB
    // ADD_SGEMM3(32, 32, 128, 2, 4, 8, false, false, 1); // best for 128
    // ADD_SGEMM3(32, 32, 128, 2, 4, 8, true, false, 1);  // best for 256
    // ADD_SGEMM3(64, 64, 64, 4, 4, 8, true, false, 4);   // best for 512
    // ADD_SGEMM3(32, 64, 64, 4, 4, 8, true, false, 1);   // best for 1024
    // ADD_SGEMM3(128, 64, 32, 8, 4, 1, true, false, 1);  // best for 2048
    // ADD_SGEMM3(64, 64, 32, 8, 4, 2, true, false, 1);   // best for 4096

#define ADD_SGEMM4(BM, BN, BK, WM, WN, WNITER, TM, TN, TK, PREFETCH_GLOBAL)                                            \
    do {                                                                                                               \
        constexpr int NUM_WARPS_M = BM / WM;                                                                           \
        constexpr int NUM_WARPS_N = BN / WN;                                                                           \
        constexpr int NUM_THREADS = NUM_WARPS_M * NUM_WARPS_N * WARP_SIZE;                                             \
        constexpr bool is_valid_num_threads = (32 <= NUM_THREADS) && (NUM_THREADS <= 1024);                            \
        constexpr bool is_shared_oom = (BN == 128 && BK == 128) || BM == 128 && BK == 128 ||                           \
                                       (BM == 64 && BN == 64 && BK == 128) || (BM == 128 && BN == 128 && BK == 64);    \
        constexpr bool is_valid_warp_tile = (WN % (WNITER * TN) == 0) && (WARP_SIZE * WNITER * TN % WN == 0) &&        \
                                            ((WM * WN) % (WARP_SIZE * WNITER * TM * TN) == 0);                         \
        constexpr bool is_valid_load_tile =                                                                            \
            is_valid_num_threads && ((BK * BN) % (NUM_THREADS * 4) == 0) && ((BM * BK) % (NUM_THREADS * 4) == 0);      \
        constexpr int WSUBN = TRY_DIV(WN, WNITER);                                                                     \
        constexpr int NUM_LANES_N = TRY_DIV(WSUBN, TN);                                                                \
        constexpr int NUM_LANES_M = TRY_DIV(WARP_SIZE, NUM_LANES_N);                                                   \
        constexpr int WSUBM = NUM_LANES_M * TM;                                                                        \
        constexpr int WMITER = TRY_DIV(WM, WSUBM);                                                                     \
        constexpr bool is_4x8_warp = (NUM_LANES_N == 4 && NUM_LANES_M == 8) || (NUM_LANES_N == 8 && NUM_LANES_M == 4); \
        if constexpr (is_valid_num_threads && !is_shared_oom && is_valid_warp_tile && is_valid_load_tile &&            \
                      is_4x8_warp && WMITER <= 2) {                                                                    \
            ADD_KERNEL((sgemm4<BM, BN, BK, WM, WN, WNITER, TM, TN, TK, PREFETCH_GLOBAL>));                             \
        }                                                                                                              \
    } while (0)

#define ADD_SGEMM4_PREFETCH_GLOBAL(BM, BN, BK, WM, WN, WNITER, TM, TN, TK)                                             \
    ADD_SGEMM4(BM, BN, BK, WM, WN, WNITER, TM, TN, TK, false);                                                         \
    ADD_SGEMM4(BM, BN, BK, WM, WN, WNITER, TM, TN, TK, true)

#define ADD_SGEMM4_TK(BM, BN, BK, WM, WN, WNITER, TM, TN)                                                              \
    ADD_SGEMM4_PREFETCH_GLOBAL(BM, BN, BK, WM, WN, WNITER, TM, TN, 1);                                                 \
    ADD_SGEMM4_PREFETCH_GLOBAL(BM, BN, BK, WM, WN, WNITER, TM, TN, 2);                                                 \
    ADD_SGEMM4_PREFETCH_GLOBAL(BM, BN, BK, WM, WN, WNITER, TM, TN, 4)

#define ADD_SGEMM4_TN(BM, BN, BK, WM, WN, WNITER, TM) ADD_SGEMM4_TK(BM, BN, BK, WM, WN, WNITER, TM, 4)

#define ADD_SGEMM4_TM(BM, BN, BK, WM, WN, WNITER)                                                                      \
    ADD_SGEMM4_TN(BM, BN, BK, WM, WN, WNITER, 1);                                                                      \
    ADD_SGEMM4_TN(BM, BN, BK, WM, WN, WNITER, 2);                                                                      \
    ADD_SGEMM4_TN(BM, BN, BK, WM, WN, WNITER, 4)

#define ADD_SGEMM4_WNITER(BM, BN, BK, WM, WN)                                                                          \
    ADD_SGEMM4_TM(BM, BN, BK, WM, WN, 1);                                                                              \
    ADD_SGEMM4_TM(BM, BN, BK, WM, WN, 2)

#define ADD_SGEMM4_WN(BM, BN, BK, WM)                                                                                  \
    ADD_SGEMM4_WNITER(BM, BN, BK, WM, 8);                                                                              \
    ADD_SGEMM4_WNITER(BM, BN, BK, WM, 16);                                                                             \
    ADD_SGEMM4_WNITER(BM, BN, BK, WM, 32);                                                                             \
    ADD_SGEMM4_WNITER(BM, BN, BK, WM, 64);                                                                             \
    ADD_SGEMM4_WNITER(BM, BN, BK, WM, 128)

#define ADD_SGEMM4_WM(BM, BN, BK)                                                                                      \
    ADD_SGEMM4_WN(BM, BN, BK, 8);                                                                                      \
    ADD_SGEMM4_WN(BM, BN, BK, 16);                                                                                     \
    ADD_SGEMM4_WN(BM, BN, BK, 32);                                                                                     \
    ADD_SGEMM4_WN(BM, BN, BK, 64);                                                                                     \
    ADD_SGEMM4_WN(BM, BN, BK, 128)

#define ADD_SGEMM4_BK(BM, BN)                                                                                          \
    ADD_SGEMM4_WM(BM, BN, 32);                                                                                         \
    ADD_SGEMM4_WM(BM, BN, 64);                                                                                         \
    ADD_SGEMM4_WM(BM, BN, 128)

#define ADD_SGEMM4_BN(BM)                                                                                              \
    ADD_SGEMM4_BK(BM, 32);                                                                                             \
    ADD_SGEMM4_BK(BM, 64);                                                                                             \
    ADD_SGEMM4_BK(BM, 128)

#define ADD_SGEMM4_ALL                                                                                                 \
    ADD_SGEMM4_BN(32);                                                                                                 \
    ADD_SGEMM4_BN(64);                                                                                                 \
    ADD_SGEMM4_BN(128)

    // ADD_SGEMM4(32, 32, 32, 32, 32, 2, 4, 4, 4, true);
    // ADD_SGEMM4_ALL;

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    struct PerfRecord {
        std::string name;
        float elapsed = INFINITY;
    };

    PerfRecord best_record;
    PerfRecord cublas_record;

    for (const auto &item : kernels) {
        const std::string &name = item.first;
        const auto fn = item.second;

        float *C1 = (float *)malloc(M * N * sizeof(float));
        float *C2 = (float *)malloc(M * N * sizeof(float));

        float *dC1;
        CHECK_CUDA(cudaMalloc(&dC1, M * N * sizeof(float)));
        float *dC2;
        CHECK_CUDA(cudaMalloc(&dC2, M * N * sizeof(float)));

        // cuda impl
        fn(M, N, K, dA, dB, dC1);
        CHECK_CUDA(cudaMemcpy(C1, dC1, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // cublas impl
        cublas_sgemm(handle, M, N, K, dA, dB, dC2);
        CHECK_CUDA(cudaMemcpy(C2, dC2, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // check correctness
        bool is_correct = true;
        for (int i = 0; i < M * N; i++) {
            if (!is_close(C1[i], C2[i], 1e-4, 1e-5)) {
                int x = i % N;
                int y = i / N;
                printf("[%s] error: result diff at (%d, %d): c1=%f vs c2=%f\n", name.c_str(), y, x, C1[i], C2[i]);
                is_correct = false;
                break;
            }
        }

        if (is_correct) {
            auto perf_fn = [=] { fn(M, N, K, dA, dB, dC1); };
            const int warmup = std::max(4096 / M, 1);
            const int active = warmup * 4;
            const float elapsed = timeit(perf_fn, warmup, active);

            const float tflops = (2ull * M * N * K) / 1e12f / elapsed;
            const float bandwidth = (M * K + K * N + M * N) * sizeof(float) / 1e9f / elapsed;

            printf("[%s] elapsed %.3f us, %.1f TFLOPS, %.3f GB/s\n", name.c_str(), elapsed * 1e6, tflops, bandwidth);

            if (name == "cublas") {
                cublas_record.name = name;
                cublas_record.elapsed = elapsed;
            } else if (elapsed < best_record.elapsed) {
                best_record.name = name;
                best_record.elapsed = elapsed;
            }
        }

        free(C1);
        free(C2);
        CHECK_CUDA(cudaFree(dC1));
        CHECK_CUDA(cudaFree(dC2));
    }

    printf("[best] %s vs cublas: %.1f%% (%.3f vs %.3f ms)\n", best_record.name.c_str(),
           cublas_record.elapsed / best_record.elapsed * 100.f, best_record.elapsed * 1e3f,
           cublas_record.elapsed * 1e3f);

    CHECK_CUBLAS(cublasDestroy(handle));

    free(A);
    free(B);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
}

int main(int argc, char **argv) {
    // both square
    {
        int dims[]{128, 256, 512, 1024, 2048, 4096};
        for (int d : dims) {
            perf(d, d, d);
        }
    }

    // non-square
    // {
    //     int dims[]{512, 1024, 2048};
    //     for (int M : dims) {
    //         for (int N : dims) {
    //             for (int K : dims) {
    //                 if (M == N && N == K) {
    //                     continue;
    //                 }
    //                 perf(M, N, K);
    //             }
    //         }
    //     }
    // }

    return 0;
}
