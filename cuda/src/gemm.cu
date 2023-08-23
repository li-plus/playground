#include "gemm.h"

#include <iostream>

__global__ void sgemm1_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {
    constexpr int BLOCK_SIZE = 32;
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < N && y < M) {
        float s = 0;
        for (int k = 0; k < K; k++) {
            s += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = s;
    }
}

__global__ void sgemm2_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {
    constexpr int BLOCK_SIZE = 32;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];

    if (x < N && y < M) {
        float s = 0;
        for (int k = 0; k < K; k += BLOCK_SIZE) {
            shA[ty][tx] = A[y * K + (k + tx)];
            shB[ty][tx] = B[(k + ty) * N + x];
            __syncthreads();
#pragma unroll
            for (int blk_k = 0; blk_k < BLOCK_SIZE; blk_k++) {
                s += shA[ty][blk_k] * shB[blk_k][tx];
            }
            __syncthreads();
        }
        C[y * N + x] = s;
    }
}

__device__ static inline float4 operator+(const float4 &a, const float4 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__ static inline float4 &operator+=(float4 &self, const float4 &other) { return self = self + other; }

__device__ static inline float4 operator*(const float4 &a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int THREAD_SIZE_X, int THREAD_SIZE_Y, bool DO_PREFETCH>
__global__ void sgemm3_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                              float *__restrict__ C) {
#define A_AT(row, col) A[(row)*K + (col)]
#define B_AT(row, col) B[(row)*N + (col)]
#define C_AT(row, col) C[(row)*N + (col)]

    static_assert(THREAD_SIZE_X == 4);

    constexpr int NUM_THREADS_X = BLOCK_SIZE_N / THREAD_SIZE_X;
    constexpr int NUM_THREADS_Y = BLOCK_SIZE_M / THREAD_SIZE_Y;
    constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;

    constexpr int A_NUM_THREADS_X = BLOCK_SIZE_K / 4;
    constexpr int A_NUM_THREADS_Y = NUM_THREADS / A_NUM_THREADS_X;
    constexpr int B_NUM_THREADS_X = BLOCK_SIZE_N / 4;
    constexpr int B_NUM_THREADS_Y = NUM_THREADS / B_NUM_THREADS_X;

    static_assert(A_NUM_THREADS_Y <= BLOCK_SIZE_M);
    static_assert(B_NUM_THREADS_Y <= BLOCK_SIZE_K);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * NUM_THREADS_X + tx;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int A_blk_x = tid % A_NUM_THREADS_X * 4;
    const int A_blk_start_y = tid / A_NUM_THREADS_X;
    const int B_blk_x = tid % B_NUM_THREADS_X * 4;
    const int B_blk_start_y = tid / B_NUM_THREADS_X;

    __shared__ float shA[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float shB[BLOCK_SIZE_K][BLOCK_SIZE_N];

    float4 sums[THREAD_SIZE_Y] = {};

    if (!DO_PREFETCH) {
        for (int k = 0; k < K; k += BLOCK_SIZE_K) {
#pragma unroll
            for (int A_blk_y = A_blk_start_y; A_blk_y < BLOCK_SIZE_M; A_blk_y += A_NUM_THREADS_Y) {
                *(float4 *)&shA[A_blk_y][A_blk_x] = *(float4 *)&A_AT(by * BLOCK_SIZE_M + A_blk_y, k + A_blk_x);
            }
#pragma unroll
            for (int B_blk_y = B_blk_start_y; B_blk_y < BLOCK_SIZE_K; B_blk_y += B_NUM_THREADS_Y) {
                *(float4 *)&shB[B_blk_y][B_blk_x] = *(float4 *)&B_AT(k + B_blk_y, bx * BLOCK_SIZE_N + B_blk_x);
            }
            __syncthreads();

#pragma unroll
            for (int blk_k = 0; blk_k < BLOCK_SIZE_K; blk_k += THREAD_SIZE_X) {
#pragma unroll
                for (int thr_y = 0; thr_y < THREAD_SIZE_Y; thr_y++) {
                    float4 thA = *(float4 *)&shA[ty * THREAD_SIZE_Y + thr_y][blk_k];
#pragma unroll
                    for (int thr_k = 0; thr_k < 4; thr_k++) {
                        float4 thB = *(float4 *)&shB[blk_k + thr_k][tx * THREAD_SIZE_X];
                        sums[thr_y] += thB * ((float *)&thA)[thr_k];
                    }
                }
            }
            __syncthreads();
        }
    } else {
        static_assert(B_NUM_THREADS_Y % THREAD_SIZE_X == 0);

        float4 next_B;

#pragma unroll
        for (int B_blk_y = B_blk_start_y; B_blk_y < BLOCK_SIZE_K; B_blk_y += B_NUM_THREADS_Y) {
            *(float4 *)&shB[B_blk_y][B_blk_x] = *(float4 *)&B_AT(B_blk_y, bx * BLOCK_SIZE_N + B_blk_x);
        }

        for (int k = 0; k < K; k += BLOCK_SIZE_K) {
#pragma unroll
            for (int A_blk_y = A_blk_start_y; A_blk_y < BLOCK_SIZE_M; A_blk_y += A_NUM_THREADS_Y) {
                *(float4 *)&shA[A_blk_y][A_blk_x] = *(float4 *)&A_AT(by * BLOCK_SIZE_M + A_blk_y, k + A_blk_x);
            }

            const int next_k = (k + BLOCK_SIZE_K < K) ? k + BLOCK_SIZE_K : 0;
            __syncthreads();

#pragma unroll
            for (int blk_k = 0; blk_k < BLOCK_SIZE_K; blk_k += THREAD_SIZE_X) {
                // prefetch global memory to register
                const int B_blk_y = B_blk_start_y + B_NUM_THREADS_Y * (blk_k / B_NUM_THREADS_Y);
                if (blk_k % B_NUM_THREADS_Y == 0) {
                    next_B = *(float4 *)&B_AT(next_k + B_blk_y, bx * BLOCK_SIZE_N + B_blk_x);
                }
                // compute
#pragma unroll
                for (int thr_y = 0; thr_y < THREAD_SIZE_Y; thr_y++) {
                    float4 thA = *(float4 *)&shA[ty * THREAD_SIZE_Y + thr_y][blk_k];
#pragma unroll
                    for (int thr_k = 0; thr_k < THREAD_SIZE_X; thr_k++) {
                        float4 thB = *(float4 *)&shB[blk_k + thr_k][tx * THREAD_SIZE_X];
                        sums[thr_y] += thB * ((float *)&thA)[thr_k];
                    }
                }
                // store to shared memory
                if ((blk_k + THREAD_SIZE_X) % B_NUM_THREADS_Y == 0) {
                    __syncthreads();
                    *(float4 *)&shB[B_blk_y][B_blk_x] = next_B;
                }
            }
        }
    }

#pragma unroll
    for (int thr_y = 0; thr_y < THREAD_SIZE_Y; thr_y++) {
        *(float4 *)&C_AT(by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + thr_y, bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X) =
            sums[thr_y];
    }

#undef A_AT
#undef B_AT
#undef C_AT
}

static inline int ceiling(int a, int b) { return (a + b - 1) / b; }

static inline cudaError_t sgemm1_cuda(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int block_size = 32;
    dim3 blocks(ceiling(N, block_size), ceiling(M, block_size));
    dim3 threads(block_size, block_size);
    sgemm1_kernel<<<blocks, threads>>>(M, N, K, A, B, C);
    return cudaSuccess;
}

static inline cudaError_t sgemm2_cuda(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int block_size = 32;
    dim3 blocks(ceiling(N, block_size), ceiling(M, block_size));
    dim3 threads(block_size, block_size);
    sgemm2_kernel<<<blocks, threads>>>(M, N, K, A, B, C);
    return cudaSuccess;
}

static inline cudaError_t sgemm3_cuda(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_SIZE_M = 32;
    constexpr int BLOCK_SIZE_N = 64;
    constexpr int BLOCK_SIZE_K = 32;
    constexpr int THREAD_SIZE_X = 4;
    constexpr int THREAD_SIZE_Y = 4;
    constexpr bool DO_PREFETCH = true;
    if (N % BLOCK_SIZE_N != 0 || M % BLOCK_SIZE_M != 0 || K % BLOCK_SIZE_K != 0) {
        fprintf(stderr, "invalid matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
        return cudaErrorInvalidValue;
    }
    dim3 blocks(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 threads(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    sgemm3_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y, DO_PREFETCH>
        <<<blocks, threads>>>(M, N, K, A, B, C);
    return cudaSuccess;
}

cudaError_t sgemm_cuda(int M, int N, int K, const float *A, const float *B, float *C) {
    return sgemm3_cuda(M, N, K, A, B, C);
}
