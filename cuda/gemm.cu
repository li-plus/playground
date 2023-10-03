#include "common.h"
#include <functional>
#include <iostream>
#include <vector>

__global__ void sgemm_1_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
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

static inline void sgemm_1(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_DIM_X = 32;
    constexpr int BLOCK_DIM_Y = 32;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM_X), ceil_div(M, BLOCK_DIM_Y));
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm_1_kernel<<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

template <int BLOCK_DIM>
__global__ void sgemm_2_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
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

static inline void sgemm_2(int M, int N, int K, const float *A, const float *B, float *C) {
    constexpr int BLOCK_DIM = 32;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM), ceil_div(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    sgemm_2_kernel<BLOCK_DIM><<<grid_dim, block_dim>>>(M, N, K, A, B, C);
}

__device__ static inline float4 operator+(const float4 &a, const float4 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__device__ static inline float4 &operator+=(float4 &self, const float4 &other) { return self = self + other; }

__device__ static inline float4 operator*(const float4 &a, float s) { return {a.x * s, a.y * s, a.z * s, a.w * s}; }

template <int BM, int BN, int BK, int TM, int TN, int TK>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_3_kernel(int M, int N, int K, const float *__restrict__ A, const float *__restrict__ B,
                   float *__restrict__ C) {
    static_assert(TM % 4 == 0 && TN % 4 == 0 && TK % 4 == 0); // float4

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tid = ty * blockDim.x + tx;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float Areg[TM * TK];
    float sums[TM * TN] = {};

    A += by * BM * K; // move A to top-left corner of first row block
    B += bx * BN;     // move B to top-left corner of first column block

    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    // constants for loading A/B from global memory into shared memory
    static_assert((NUM_THREADS * 4) % BK == 0 || BK % (NUM_THREADS * 4));
    constexpr int A_LOAD_TILE_Y = (NUM_THREADS * 4 < BK) ? 1 : NUM_THREADS * 4 / BK;
    constexpr int A_LOAD_TILE_X = (NUM_THREADS * 4 < BK) ? NUM_THREADS * 4 : BK;
    static_assert(BM % A_LOAD_TILE_Y == 0);
    const int A_y_offset = tid * 4 / BK;
    const int A_x_offset = tid * 4 % BK;

    // static_assert((BK * BN) % (NUM_THREADS * 4) == 0);
    static_assert((NUM_THREADS * 4) % BN == 0 || BN % (NUM_THREADS * 4) == 0);
    constexpr int B_LOAD_TILE_Y = (NUM_THREADS * 4 < BN) ? 1 : NUM_THREADS * 4 / BN;
    constexpr int B_LOAD_TILE_X = (NUM_THREADS * 4 < BN) ? NUM_THREADS * 4 : BN;
    static_assert(BK % B_LOAD_TILE_Y == 0);
    const int B_y_offset = tid * 4 / BN;
    const int B_x_offset = tid * 4 % BN;

    for (int k = 0; k < K; k += BK) {
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
        __syncthreads();

        float *pAs = As + ty * TM * BK;
        float *pBs = Bs + tx * TN;

#pragma unroll
        for (int bk = 0; bk < BK; bk += TK) {
            // each thread loads pAs[0:TM][0:TK] into Areg
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tk = 0; tk < TK; tk += 4) {
                    *(float4 *)&Areg[tm * TK + tk] = *(float4 *)&pAs[tm * BK + tk];
                }
            }
            // each thread loads pBs[0:TK][0:TN] into Breg and compute matmul
#pragma unroll
            for (int tk = 0; tk < TK; tk++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn += 4) {
                    float4 Breg = *(float4 *)&pBs[tk * BN + tn];
#pragma unroll
                    for (int tm = 0; tm < TM; tm += 4) {
                        *(float4 *)&sums[(tm + 0) * TN + tn] += Breg * Areg[(tm + 0) * TK + tk];
                        *(float4 *)&sums[(tm + 1) * TN + tn] += Breg * Areg[(tm + 1) * TK + tk];
                        *(float4 *)&sums[(tm + 2) * TN + tn] += Breg * Areg[(tm + 2) * TK + tk];
                        *(float4 *)&sums[(tm + 3) * TN + tn] += Breg * Areg[(tm + 3) * TK + tk];
                    }
                }
            }

            pAs += TK;
            pBs += TK * BN;
        }

        A += BK;
        B += BK * N;

        __syncthreads();
    }

#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn += 4) {
            *(float4 *)&C[(by * BM + ty * TM + tm) * N + bx * BN + tx * TN + tn] = *(float4 *)&sums[tm * TN + tn];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int TM = 4, int TN = 4, int TK = 4>
static inline void sgemm_3(int M, int N, int K, const float *A, const float *B, float *C) {
    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "invalid matrix dimensions";

    static_assert(BM % TM == 0 && BN % TN == 0 && BK % TK == 0);

    constexpr int BLOCK_DIM_X = BN / TN;
    constexpr int BLOCK_DIM_Y = BM / TM;
    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024);

    dim3 grid_dim(N / BN, M / BM);
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm_3_kernel<BM, BN, BK, TM, TN, TK><<<grid_dim, block_dim>>>(M, N, K, A, B, C);
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
            {"sgemm_1", sgemm_1},
            {"sgemm_2", sgemm_2},
            {"cublas", [handle](int M, int N, int K, const float *dA, const float *dB,
                                float *dC) { cublas_sgemm(handle, M, N, K, dA, dB, dC); }},
        };

#define ADD_KERNEL(stmt) kernels.emplace_back(#stmt, (stmt))
#define ADD_SGEMM3(BM, BN, BK, TM, TN, TK)                                                                             \
    do {                                                                                                               \
        if constexpr (!(((BN) == 128 && (BK) == 128) || ((BM) == 128 && (BK) == 128) ||                                \
                        ((BM) == 64 && (BN) == 64 && (BK) == 128) || ((BM) == 128 && (BN) == 128 && (BK) == 64) ||     \
                        ((BM) / (TM) * (BN) / (TN) < 32))) {                                                           \
            ADD_KERNEL((sgemm_3<(BM), (BN), (BK), (TM), (TN), (TK)>));                                                 \
        }                                                                                                              \
    } while (0)
#define ADD_SGEMM3_BM(BN, BK, TM, TN, TK)                                                                              \
    ADD_SGEMM3(32, BN, BK, TM, TN, TK);                                                                                \
    ADD_SGEMM3(64, BN, BK, TM, TN, TK);                                                                                \
    ADD_SGEMM3(128, BN, BK, TM, TN, TK)
#define ADD_SGEMM3_BN(BK, TM, TN, TK)                                                                                  \
    ADD_SGEMM3_BM(32, BK, TM, TN, TK);                                                                                 \
    ADD_SGEMM3_BM(64, BK, TM, TN, TK);                                                                                 \
    ADD_SGEMM3_BM(128, BK, TM, TN, TK)
#define ADD_SGEMM3_BK(TM, TN, TK)                                                                                      \
    ADD_SGEMM3_BN(32, TM, TN, TK);                                                                                     \
    ADD_SGEMM3_BN(64, TM, TN, TK);                                                                                     \
    ADD_SGEMM3_BN(128, TM, TN, TK)
#define ADD_SGEMM3_TM(TN, TK)                                                                                          \
    ADD_SGEMM3_BK(4, TN, TK);                                                                                          \
    ADD_SGEMM3_BK(8, TN, TK)
#define ADD_SGEMM3_TN(TK)                                                                                              \
    ADD_SGEMM3_TM(4, TK);                                                                                              \
    ADD_SGEMM3_TM(8, TK)
#define ADD_SGEMM3_ALL                                                                                                 \
    ADD_SGEMM3_TN(4);                                                                                                  \
    ADD_SGEMM3_TN(8)

    ADD_SGEMM3_ALL;

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    struct PerfRecord {
        std::string name;
        float elapsed_ms = INFINITY;
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
                printf("[%s] error: value diff at (%d, %d): c1=%f vs c2=%f\n", name.c_str(), y, x, C1[i], C2[i]);
                is_correct = false;
                break;
            }
        }

        if (!is_correct) {
            continue;
        }

        auto perf_fn = [=] { fn(M, N, K, dA, dB, dC1); };
        float elapsed_ms = timeit(perf_fn, 3, 10);

        float tflops_peak = V100SXM2Spec::PEAK_FP32_TFLOPS;
        float tflops_actual = (2ull * M * N * K) / elapsed_ms / (1000.f * 1000.f * 1000.f);
        float mfu = tflops_actual / tflops_peak;

        float bw_peak = V100SXM2Spec::PEAK_MEM_BW;
        float bw_actual = (M * K + K * N + M * N) * sizeof(float) / (float)GB / (elapsed_ms / 1000);
        float mbu = bw_actual / bw_peak;

        printf("[%s] elapsed %.3f ms, mfu %.3f (%.1f/%.1f TFLOPS), mbu %.3f (%.1f/%.1f GB/s)\n", name.c_str(),
               elapsed_ms, mfu, tflops_actual, tflops_peak, mbu, bw_actual, bw_peak);

        if (name == "cublas") {
            cublas_record.name = name;
            cublas_record.elapsed_ms = elapsed_ms;
        } else if (elapsed_ms < best_record.elapsed_ms) {
            best_record.name = name;
            best_record.elapsed_ms = elapsed_ms;
        }

        free(C1);
        free(C2);
        CHECK_CUDA(cudaFree(dC1));
        CHECK_CUDA(cudaFree(dC2));
    }

    printf("[best] %s vs cublas: %.1f%%\n", best_record.name.c_str(),
           cublas_record.elapsed_ms / best_record.elapsed_ms * 100.f);

    CHECK_CUBLAS(cublasDestroy(handle));

    free(A);
    free(B);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
}

int main(int argc, char **argv) {
    int dims[]{128, 256, 512, 1024, 2048, 4096};
    for (int d : dims) {
        perf(d, d, d);
    }
    return 0;
}
