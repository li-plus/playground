#include "common.h"
#include <functional>
#include <iostream>
#include <vector>

__global__ void sgemm_1_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
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

__global__ void sgemm_2_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
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
__global__ void sgemm_3_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                               float *__restrict__ C) {
#define A_AT(row, col) A[(row) * K + (col)]
#define B_AT(row, col) B[(row) * N + (col)]
#define C_AT(row, col) C[(row) * N + (col)]

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

static inline void sgemm_1(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int block_size = 32;
    dim3 blocks(ceil_div(N, block_size), ceil_div(M, block_size));
    dim3 threads(block_size, block_size);
    sgemm_1_kernel<<<blocks, threads>>>(M, N, K, A, B, C);
}

static inline void sgemm_2(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int block_size = 32;
    dim3 blocks(ceil_div(N, block_size), ceil_div(M, block_size));
    dim3 threads(block_size, block_size);
    sgemm_2_kernel<<<blocks, threads>>>(M, N, K, A, B, C);
}

static inline void sgemm_3(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_SIZE_M = 32;
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int BLOCK_SIZE_K = 32;
    constexpr int THREAD_SIZE_X = 4;
    constexpr int THREAD_SIZE_Y = 4;
    constexpr bool DO_PREFETCH = true;

    CHECK(N % BLOCK_SIZE_N == 0 && M % BLOCK_SIZE_M == 0 && K % BLOCK_SIZE_K == 0) << "invalid matrix dimensions";

    dim3 blocks(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    dim3 threads(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    sgemm_3_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y, DO_PREFETCH>
        <<<blocks, threads>>>(M, N, K, A, B, C);
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

    float *C1 = (float *)malloc(M * N * sizeof(float));

    float *C2 = (float *)malloc(M * N * sizeof(float));

    float *dA;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));

    float *dB;
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float *dC1;
    CHECK_CUDA(cudaMalloc(&dC1, M * N * sizeof(float)));

    float *dC2;
    CHECK_CUDA(cudaMalloc(&dC2, M * N * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::vector<std::pair<std::string, std::function<void(int, int, int, const float *, const float *, float *)>>>
        kernels{
            {"sgemm_1", sgemm_1},
            {"sgemm_2", sgemm_2},
            {"sgemm_3", sgemm_3},
            {"cublas", [handle](int M, int N, int K, const float *dA, const float *dB,
                                float *dC) { cublas_sgemm(handle, M, N, K, dA, dB, dC); }},
        };

    for (const auto &item : kernels) {
        const std::string &name = item.first;
        const auto fn = item.second;

        // cuda impl
        fn(M, N, K, dA, dB, dC1);
        CHECK_CUDA(cudaMemcpy(C1, dC1, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // cublas impl
        cublas_sgemm(handle, M, N, K, dA, dB, dC2);
        CHECK_CUDA(cudaMemcpy(C2, dC2, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // check correctness
        for (int i = 0; i < M * N; i++) {
            if (!is_close(C1[i], C2[i], 1e-4, 1e-5)) {
                int x = i % N;
                int y = i / N;
                printf("value diff at (%d, %d): c1=%f vs c2=%f\n", y, x, C1[i], C2[i]);
                break;
            }
        }

        auto perf_fn = [=] { fn(M, N, K, dA, dB, dC1); };
        float elapsed_ms = timeit(perf_fn, 3, 10);

        float tflops_peak = V100SXM2Spec::PEAK_FP32_TFLOPS;
        float tflops_actual = (2ull * M * N * (K - 1)) / elapsed_ms / (1000.f * 1000.f * 1000.f);
        float mfu = tflops_actual / tflops_peak;

        float bw_peak = V100SXM2Spec::PEAK_MEM_BW;
        float bw_actual = (M * K + K * N + M * N) * sizeof(float) / (float)GB / (elapsed_ms / 1000);
        float mbu = bw_actual / bw_peak;

        printf("[%8s] M=%4d N=%4d K=%4d elapsed %.3f ms, mfu %.3f (%.1f/%.1f TFLOPS), mbu %.3f (%.1f/%.1f GB/s)\n",
               name.c_str(), M, N, K, elapsed_ms, mfu, tflops_actual, tflops_peak, mbu, bw_actual, bw_peak);
    }

    CHECK_CUBLAS(cublasDestroy(handle));

    free(A);
    free(B);
    free(C1);
    free(C2);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC1));
    CHECK_CUDA(cudaFree(dC2));
}

int main(int argc, char **argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    perf(M, N, K);
    return 0;
}
