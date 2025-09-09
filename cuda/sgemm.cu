// Tutorial:
// https://siboehm.com/articles/22/CUDA-MMM
// https://zhuanlan.zhihu.com/p/657632577
// CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

#include "common.h"
#include <functional>
#include <vector>

// #define SGEMM_DEBUG

// naive kernel
__global__ void sgemm_v1_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M,
                                int N, int K) {
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

void sgemm_v1(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM_X), ceil_div(M, BLOCK_DIM_Y));
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm_v1_kernel<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

// using shared memory
template <int BLOCK_DIM>
__global__ void sgemm_v2_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M,
                                int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float s_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_B[BLOCK_DIM][BLOCK_DIM];

    A += by * BLOCK_DIM * K;
    B += bx * BLOCK_DIM;
    C += (by * N + bx) * BLOCK_DIM;

    float s = 0;
    for (int bk = 0; bk < K; bk += BLOCK_DIM) {
        s_A[ty][tx] = A[ty * K + tx];
        s_B[ty][tx] = B[ty * N + tx];
        __syncthreads();
#pragma unroll
        for (int tk = 0; tk < BLOCK_DIM; tk++) {
            s += s_A[ty][tk] * s_B[tk][tx];
        }
        A += BLOCK_DIM;
        B += BLOCK_DIM * N;
        __syncthreads();
    }
    C[ty * N + tx] = s;
}

void sgemm_v2(const float *A, const float *B, float *C, int M, int N, int K) {
    constexpr int BLOCK_DIM = 16;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM), ceil_div(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    sgemm_v2_kernel<BLOCK_DIM><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_v3_kernel(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N,
                    int K) {

    /*
block tiling:
                                  BN
                            +===========+
                            |   |   |   |
                            |   +---+   |
                         BK |   |   |   |   Matrix B
                            |   +---+   |
                            |   |   |   |
                            +===========+
             BK
    +===================+   +===========+
    |                   |   |           |
    |-------+---+-------|   |   +---+   |
 BM |       |   |       |   |   |   |   |
    |-------+---+-------|   |   +---+   |
    |                   |   |           |
    +===================+   +===========+
           Matrix A            Matrix C

thread tiling:
                                  TN
                            +===========+
                            |   |   |   |
                            |   +---+   |
                       TK=1 |   |   |   |   Block B
                            |   +---+   |
                            |   |   |   |
                            +===========+
             TK=1
    +===================+   +===========+
    |                   |   |           |
    |-------+---+-------|   |   +---+   |
 TM |       |   |       |   |   |   |   |
    |-------+---+-------|   |   +---+   |
    |                   |   |           |
    +===================+   +===========+
           Block A             Block C

Each thread handles TM * TN elements of C. When TM > 4 or TN > 4, one thread needs to load adjacent elements more than
16 bytes sequentially, causing bank conflict. To avoid this, we split the thread tile into TM/4 x TN/4 sub-tiles. In
each sub-tiles, one thread only handles 16 bytes at a time.

                                  4       4
                            +===================+
                            |   |   |   |   |   |
                            |   +---+   +---+   |
                            |   +---+   +---+   |   Block B
                            |   |   |   |   |   |
                            |   |   |   |   |   |
                            +===================+

    +===================+   +===================+
    |                   |   |                   |
  4 |---+---+-----------|   |   +---+   +---+   |
    |---+---+-----------|   |   +---+   +---+   |
    |                   |   |                   |
  4 |---+---+-----------|   |   +---+   +---+   |
    |---+---+-----------|   |   +---+   +---+   |
    |                   |   |                   |
    +===================+   +===================+
           Block A                 Block C
    */

    constexpr int BX = BN / TN; // blockDim.x
    constexpr int BY = BM / TM; // blockDim.y
    constexpr int NUM_THREADS = BY * BX;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    const float *A_block = A + by * BM * K;
    const float *B_block = B + bx * BN;
    float *C_block = C + by * BM * N + bx * BN;

    float sums[TM][TN]{};

    for (int k = 0; k < K; k += BK) {
        // load BM * BK tile of A into shared memory
        {
            static_assert((BM * BK) % (NUM_THREADS * 4) == 0, "unimplemented: corrupted load of A");
            static_assert(BK <= NUM_THREADS * 4, "unimplemented: BK is too large");
            constexpr int A_LOAD_TILE_X = BK / 4;
            constexpr int A_LOAD_TILE_Y = NUM_THREADS / A_LOAD_TILE_X;
            const int x = tid % A_LOAD_TILE_X * 4;
#pragma unroll
            for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
                const int y = y_start + tid / A_LOAD_TILE_X;
                *(float4 *)&s_A[y][x] = *(float4 *)&A_block[y * K + x];
            }
        }

        // load BK * BN tile of B into shared memory
        {
            static_assert((BK * BN) % (NUM_THREADS * 4) == 0, "unimplemented: corrupted load of B");
            static_assert(BN <= NUM_THREADS * 4, "unimplemented: BN is too large");
            constexpr int B_LOAD_TILE_X = BN / 4;
            constexpr int B_LOAD_TILE_Y = NUM_THREADS / B_LOAD_TILE_X;
            const int x = tid % B_LOAD_TILE_X * 4;
#pragma unroll
            for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
                const int y = y_start + tid / B_LOAD_TILE_X;
                *(float4 *)&s_B[y][x] = *(float4 *)&B_block[y * N + x];
            }
        }

        __syncthreads();

#ifdef SGEMM_DEBUG
        if (bx == 0 && by == 0 && tid == 0) {
            printf("===== block (%d, %d), tid (%d, %d), k=%d =====\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
                   k);
            printf("s_A:\n");
            for (int i = 0; i < BM; i++) {
                for (int j = 0; j < BK; j++) {
                    printf("%.2f, ", s_A[i][j]);
                }
                printf("\n");
            }
            printf("s_B:\n");
            for (int i = 0; i < BK; i++) {
                for (int j = 0; j < BN; j++) {
                    printf("%.2f, ", s_B[i][j]);
                }
                printf("\n");
            }
        }
#endif

        static_assert(TN % 4 == 0, "unimplemented: TN is not multiple of 4");

        float reg_A[TM];
        float reg_B[TN];

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // load s_A tile into reg_A
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                reg_A[tm] = s_A[ty * TM + tm][tk]; // bank conflict
            }

            // load s_B tile into reg_B
            // if TN > 4, split into sub-tiles to avoid bank conflict
#pragma unroll
            for (int tn = 0; tn < TN; tn += 4) {
                *(float4 *)&reg_B[tn] = *(float4 *)&s_B[tk][tn * BX + tx * 4];
            }

            // outer product
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    sums[tm][tn] += reg_A[tm] * reg_B[tn];
                }
            }
        }

        __syncthreads();

        A_block += BK;
        B_block += BK * N;
    }

    // store sums to C
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn += 4) {
            *(float4 *)&C_block[(ty * TM + tm) * N + tn * BX + tx * 4] = *(float4 *)&sums[tm][tn];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int TM = 4, int TN = 4>
void sgemm_v3(const float *A, const float *B, float *C, int M, int N, int K) {
    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "invalid matrix dimensions";

    static_assert(BM % TM == 0 && BN % TN == 0);

    constexpr int BLOCK_DIM_X = BN / TN;
    constexpr int BLOCK_DIM_Y = BM / TM;
    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024);

    dim3 grid_dim(N / BN, M / BM);
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    sgemm_v3_kernel<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

void sgemm_cublas(cublasHandle_t handle, const float *A, const float *B, float *C, int M, int N, int K) {
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

void perf(int M, int N, int K) {
    // make data
    float *A, *B;
    CHECK_CUDA(cudaMallocHost(&A, sizeof(float) * M * K));
    CHECK_CUDA(cudaMallocHost(&B, sizeof(float) * K * N));

    for (int i = 0; i < M * K; i++) {
#ifndef SGEMM_DEBUG
        A[i] = uniform();
#else
        A[i] = i / 100.f;
#endif
    }

    for (int i = 0; i < K * N; i++) {
#ifndef SGEMM_DEBUG
        B[i] = uniform();
#else
        B[i] = i / 100.f;
#endif
    }

    float *dA;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));

    float *dB;
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

#define SGEMM_V3_ITEM(BM, BN, BK, TM, TN)                                                                              \
    {"sgemm_v3_b" #BM "x" #BN "x" #BK "_t" #TM "x" #TN, sgemm_v3<BM, BN, BK, TM, TN>}

    std::vector<std::tuple<std::string, std::function<void(const float *, const float *, float *, int, int, int)>>>
        kernels{
            {"cublas", [handle](const float *A, const float *B, float *C, int M, int N,
                                int K) { sgemm_cublas(handle, A, B, C, M, N, K); }},
            // {"sgemm_v1", sgemm_v1},
            // {"sgemm_v2", sgemm_v2},
            SGEMM_V3_ITEM(32, 32, 16, 4, 4),
            SGEMM_V3_ITEM(32, 32, 32, 4, 4),
            SGEMM_V3_ITEM(64, 64, 16, 4, 4),
            SGEMM_V3_ITEM(64, 64, 32, 4, 4),
            SGEMM_V3_ITEM(128, 128, 8, 8, 8),
        };

#undef SGEMM_V3_ITEM

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    struct PerfRecord {
        std::string name;
        float elapsed = INFINITY;
    };

    PerfRecord best_record;
    PerfRecord cublas_record;

    for (const auto &item: kernels) {
        const auto &name = std::get<0>(item);
        const auto &fn = std::get<1>(item);

        float *C1, *C2;
        CHECK_CUDA(cudaMallocHost(&C1, M * N * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&C2, M * N * sizeof(float)));

        float *dC1, *dC2;
        CHECK_CUDA(cudaMalloc(&dC1, M * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dC2, M * N * sizeof(float)));

        // cuda impl
        fn(dA, dB, dC1, M, N, K);
        CHECK_CUDA(cudaMemcpy(C1, dC1, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        // cublas impl
        sgemm_cublas(handle, dA, dB, dC2, M, N, K);
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
            auto perf_fn = [=] { fn(dA, dB, dC1, M, N, K); };
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

        CHECK_CUDA(cudaFreeHost(C1));
        CHECK_CUDA(cudaFreeHost(C2));
        CHECK_CUDA(cudaFree(dC1));
        CHECK_CUDA(cudaFree(dC2));
    }

    printf("[best] %s vs cublas: %.1f%% (%.3f vs %.3f ms)\n", best_record.name.c_str(),
           cublas_record.elapsed / best_record.elapsed * 100.f, best_record.elapsed * 1e3f,
           cublas_record.elapsed * 1e3f);

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
}

int main(int argc, char **argv) {
    // square matrix
    {
        const int dims[]{128, 256, 512, 1024, 2048, 4096};
        for (int d : dims) {
            perf(d, d, d);
        }
    }

    // fixed K to avoid split-K kernels
    {
        constexpr int K = 1024;
        const int dims[]{1024, 2048, 4096, 8192, 16384, 32768};
        for (int d : dims) {
            perf(d, d, K);
        }
    }

    return 0;
}
