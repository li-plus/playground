// Tutorial:
// https://siboehm.com/articles/22/CUDA-MMM
// https://zhuanlan.zhihu.com/p/657632577
// CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

#include "common.h"
#include <cuda/pipeline>
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

template <int BM, int BN, int BK, int TM, int TN, int STAGES>
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

    static_assert(TN % 4 == 0, "unimplemented: TN is not multiple of 4");

    constexpr int BX = BN / TN; // blockDim.x
    constexpr int BY = BM / TM; // blockDim.y
    constexpr int NUM_THREADS = BY * BX;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BX + tx;

    extern __shared__ float smem[];

    auto s_A = (float(*)[BM][BK]) & smem[0];                // [STAGES][BM][BK]
    auto s_B = (float(*)[BK][BN]) & smem[STAGES * BM * BK]; // [STAGES][BK][BN]

    float sums[TM][TN]{};

    auto pipeline = cuda::make_pipeline();

    // utils for global -> shared
    static_assert((BM * BK) % (NUM_THREADS * 4) == 0, "unimplemented: corrupted load of A");
    static_assert(BK <= NUM_THREADS * 4, "unimplemented: BK is too large");
    constexpr int A_LOAD_TILE_Y = NUM_THREADS * 4 / BK;
    const int A_x = tid * 4 % BK;
    const int A_y = tid * 4 / BK;

    static_assert((BK * BN) % (NUM_THREADS * 4) == 0, "unimplemented: corrupted load of B");
    static_assert(BN <= NUM_THREADS * 4, "unimplemented: BN is too large");
    constexpr int B_LOAD_TILE_Y = NUM_THREADS * 4 / BN;
    const int B_x = tid * 4 % BN;
    const int B_y = tid * 4 / BN;

    auto fetch_A_block = [&](int k) {
        const int stage = k % STAGES;
        const float *A_block = A + by * BM * K + k * BK;

        // load BM * BK tile of A into shared memory
#pragma unroll
        for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
            const int y = y_start + A_y;
            cuda::memcpy_async((float4 *)&s_A[stage][y][A_x], (float4 *)&A_block[y * K + A_x], sizeof(float4),
                               pipeline);
        }
    };

    auto fetch_B_block = [&](int k) {
        const int stage = k % STAGES;
        const float *B_block = B + bx * BN + k * BK * N;

        // load BK * BN tile of B into shared memory
#pragma unroll
        for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
            const int y = y_start + B_y;
            cuda::memcpy_async((float4 *)&s_B[stage][y][B_x], (float4 *)&B_block[y * N + B_x], sizeof(float4),
                               pipeline);
        }
    };

    auto mma_compute = [&](int k) {
        const int stage = k % STAGES;

        float reg_A[TM];
        float reg_B[TN];

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // load s_A tile into reg_A
#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                reg_A[tm] = s_A[stage][ty * TM + tm][tk]; // bank conflict
            }

            // load s_B tile into reg_B
            // if TN > 4, split into sub-tiles to avoid bank conflict
#pragma unroll
            for (int tn = 0; tn < TN; tn += 4) {
                *(float4 *)&reg_B[tn] = *(float4 *)&s_B[stage][tk][tn * BX + tx * 4];
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
    };

#pragma unroll
    for (int k = 0; k < STAGES - 1; k++) {
        pipeline.producer_acquire();
        fetch_A_block(k);
        fetch_B_block(k);
        pipeline.producer_commit();
    }

    for (int k = STAGES - 1; k < K / BK; k++) {
        if constexpr (STAGES > 1) {
            pipeline.consumer_wait();
        }
        __syncthreads();

        pipeline.producer_acquire();
        fetch_A_block(k);
        fetch_B_block(k);
        pipeline.producer_commit();

        if constexpr (STAGES == 1) {
            pipeline.consumer_wait();
            __syncthreads();
        }

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

        mma_compute(k - (STAGES - 1));

        pipeline.consumer_release();
    }

    cuda::device::__pipeline_consumer_wait<0>(pipeline);
    __syncthreads();

#pragma unroll
    for (int i = -(STAGES - 1); i < 0; i++) {
        const int k = K / BK + i;
        mma_compute(k);
    }

    // store sums to C
    float *C_block = C + by * BM * N + bx * BN;
#pragma unroll
    for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn += 4) {
            *(float4 *)&C_block[(ty * TM + tm) * N + tn * BX + tx * 4] = *(float4 *)&sums[tm][tn];
        }
    }
}

template <int BM = 32, int BN = 32, int BK = 32, int TM = 4, int TN = 4, int STAGES = 1>
void sgemm_v3(const float *A, const float *B, float *C, int M, int N, int K) {
    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "invalid matrix dimensions";

    static_assert(BM % TM == 0 && BN % TN == 0);

    constexpr int BLOCK_DIM_X = BN / TN;
    constexpr int BLOCK_DIM_Y = BM / TM;
    constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024);

    dim3 grid_dim(N / BN, M / BM);
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);

    auto kernel_fn = sgemm_v3_kernel<BM, BN, BK, TM, TN, STAGES>;
    constexpr int smem_size = STAGES * (BM + BN) * BK * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel_fn<<<grid_dim, block_dim, smem_size>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

void sgemm_cublas(cublasHandle_t handle, const float *A, const float *B, float *C, int M, int N, int K) {
    const float alpha = 1;
    const float beta = 0;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

struct PerfRecord {
    int M;
    int N;
    int K;
    std::string name;
    float elapsed;
    float tflops;

    PerfRecord() = default;

    PerfRecord(int M, int N, int K, std::string name, float elapsed, float tflops)
        : M(M), N(N), K(K), name(std::move(name)), elapsed(elapsed), tflops(tflops) {}
};

std::vector<PerfRecord> perf(int M, int N, int K) {
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

    float *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

#define MAKE_ITEM(...) {#__VA_ARGS__, __VA_ARGS__}

    std::vector<std::tuple<std::string, std::function<void(const float *, const float *, float *, int, int, int)>>>
        kernels{
            {"cublas", [handle](const float *A, const float *B, float *C, int M, int N,
                                int K) { sgemm_cublas(handle, A, B, C, M, N, K); }},
            // {"sgemm_v1", sgemm_v1},
            // {"sgemm_v2", sgemm_v2},

            MAKE_ITEM(sgemm_v3<64, 64, 32, 4, 4>),
            MAKE_ITEM(sgemm_v3<64, 64, 64, 4, 8>),
            MAKE_ITEM(sgemm_v3<64, 64, 32, 8, 4>),
            MAKE_ITEM(sgemm_v3<64, 64, 32, 4, 4, 2>),
            MAKE_ITEM(sgemm_v3<64, 64, 64, 4, 8, 2>),
            MAKE_ITEM(sgemm_v3<64, 64, 32, 8, 4, 2>),
        };

#undef MAKE_ITEM

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    std::vector<PerfRecord> records;

    float *dC_ref;
    CHECK_CUDA(cudaMalloc(&dC_ref, M * N * sizeof(float)));

    sgemm_cublas(handle, dA, dB, dC_ref, M, N, K);

    for (const auto &item : kernels) {
        const auto &name = std::get<0>(item);
        const auto &fn = std::get<1>(item);

        float *dC_opt;
        CHECK_CUDA(cudaMalloc(&dC_opt, M * N * sizeof(float)));

        fn(dA, dB, dC_opt, M, N, K);

        check_is_close_d(dC_ref, dC_opt, M * N, 1e-4, 1e-5);

        auto perf_fn = [=] { fn(dA, dB, dC_opt, M, N, K); };
        const int warmup = std::max(4096 / M, 1);
        const int active = warmup * 4;
        const float elapsed = timeit(perf_fn, warmup, active);

        const float tflops = (2ull * M * N * K) / 1e12f / elapsed;
        const float bandwidth = (M * K + K * N + M * N) * sizeof(float) / 1e9f / elapsed;

        printf("[%s] elapsed %.3f us, %.1f TFLOPS, %.3f GB/s\n", name.c_str(), elapsed * 1e6, tflops, bandwidth);

        records.emplace_back(PerfRecord(M, N, K, name, elapsed, tflops));

        CHECK_CUDA(cudaFree(dC_opt));
    }

    auto cublas_record_it =
        std::find_if(records.begin(), records.end(), [](const PerfRecord &r) { return r.name == "cublas"; });
    CHECK(cublas_record_it != records.end());
    PerfRecord cublas_record = std::move(*cublas_record_it);
    records.erase(cublas_record_it);

    CHECK(!records.empty());
    PerfRecord best_record = *std::min_element(
        records.begin(), records.end(), [](const PerfRecord &a, const PerfRecord &b) { return a.elapsed < b.elapsed; });

    printf("[best] %s vs cublas: %.1f%% (%.1f vs %.1f TFLOPS)\n", best_record.name.c_str(),
           cublas_record.elapsed / best_record.elapsed * 100.f, best_record.tflops, cublas_record.tflops);

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_ref));

    records.emplace_back(std::move(cublas_record));
    return records;
}

void save_result(const char *save_path, const std::vector<PerfRecord> &all_records) {
    // write to csv
    FILE *stream = fopen(save_path, "w");
    fprintf(stream, "M|N|K|name|elapsed|TFLOPS\n");
    for (const auto &r : all_records) {
        fprintf(stream, "%d|%d|%d|%s|%f|%f\n", r.M, r.N, r.K, r.name.c_str(), r.elapsed, r.tflops);
    }
    fclose(stream);
}

int main(int argc, char **argv) {
    // square matrix
    {
        std::vector<PerfRecord> all_records;
        const int dims[]{1024, 2048, 3072, 4096, 6144, 8192};
        for (int d : dims) {
            auto records = perf(d, d, d);
            all_records.insert(all_records.end(), records.begin(), records.end());
        }

        save_result("sgemm_bench_square.csv", all_records);
    }

    // fixed K to avoid split-K kernels
    {
        std::vector<PerfRecord> all_records;
        constexpr int K = 1024;
        const int dims[]{1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
        for (int d : dims) {
            auto records = perf(d, d, K);
            all_records.insert(all_records.end(), records.begin(), records.end());
        }

        save_result("sgemm_bench_fixk.csv", all_records);
    }

    return 0;
}
