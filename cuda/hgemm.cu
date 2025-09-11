// Tutorial:
// https://siboehm.com/articles/22/CUDA-MMM
// https://zhuanlan.zhihu.com/p/657632577
// CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

#include "common.h"
#include <cuda_fp16.h>
#include <functional>
#include <mma.h>

// #define HGEMM_DEBUG

using namespace nvcuda;

template <typename T>
__device__ __forceinline__ void cp_async_cg(T *dst, const T *src) {
    uint32_t smem_int_ptr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_int_ptr), "l"(src), "n"(sizeof(T)));
}

__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

// shared memory
template <int BLOCK_DIM>
__global__ void hgemm_v1_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M,
                                int N, int K) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ half s_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ half s_B[BLOCK_DIM][BLOCK_DIM];

    A += by * BLOCK_DIM * K;
    B += bx * BLOCK_DIM;
    C += (by * N + bx) * BLOCK_DIM;

    float s = 0.f;
    for (int bk = 0; bk < K; bk += BLOCK_DIM) {
        s_A[ty][tx] = A[ty * K + tx];
        s_B[ty][tx] = B[ty * N + tx];
        __syncthreads();
#pragma unroll
        for (int tk = 0; tk < BLOCK_DIM; tk++) {
            s += __half2float(s_A[ty][tk]) * __half2float(s_B[tk][tx]);
        }
        A += BLOCK_DIM;
        B += BLOCK_DIM * N;
        __syncthreads();
    }
    C[ty * N + tx] = __float2half(s);
}

void hgemm_v1(const half *A, const half *B, half *C, int M, int N, int K) {
    constexpr int BLOCK_DIM = 16;
    dim3 grid_dim(ceil_div(N, BLOCK_DIM), ceil_div(M, BLOCK_DIM));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    hgemm_v1_kernel<BLOCK_DIM><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <int M, int N, int K>
__device__ __forceinline__ wmma::fragment<wmma::accumulator, M, N, K, half>
fragment_to_half(const wmma::fragment<wmma::accumulator, M, N, K, float> &frag) {
    wmma::fragment<wmma::accumulator, M, N, K, half> frag_fp16;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        *(half2 *)&frag_fp16.x[i] = __float22half2_rn(*(float2 *)&frag.x[i]);
    }
    return frag_fp16;
}

// wmma
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void hgemm_v2_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M,
                                int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.f);

    const half *A_block = A + by * WMMA_M * K;
    const half *B_block = B + bx * WMMA_N;
    half *C_block = C + by * WMMA_M * N + bx * WMMA_N;

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A_block, K);
        wmma::load_matrix_sync(b_frag, B_block, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        A_block += WMMA_K;
        B_block += WMMA_K * N;
    }

    auto c_frag_fp16 = fragment_to_half(c_frag);
    wmma::store_matrix_sync(C_block, c_frag_fp16, N, wmma::mem_row_major);

#ifdef HGEMM_DEBUG
    printf("block (%d, %d) thread (%d, %d) a_frag(%d): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] b_frag(%d): "
           "[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] C_frag(%d): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, "
           "%.2f]\n",
           bx, by, threadIdx.x, threadIdx.y, a_frag.num_storage_elements, __half2float(a_frag.x[0]),
           __half2float(a_frag.x[1]), __half2float(a_frag.x[2]), __half2float(a_frag.x[3]), __half2float(a_frag.x[4]),
           __half2float(a_frag.x[5]), __half2float(a_frag.x[6]), __half2float(a_frag.x[7]), b_frag.num_storage_elements,
           __half2float(b_frag.x[0]), __half2float(b_frag.x[1]), __half2float(b_frag.x[2]), __half2float(b_frag.x[3]),
           __half2float(b_frag.x[4]), __half2float(b_frag.x[5]), __half2float(b_frag.x[6]), __half2float(b_frag.x[7]),
           c_frag.num_storage_elements, c_frag.x[0], c_frag.x[1], c_frag.x[2], c_frag.x[3], c_frag.x[4], c_frag.x[5],
           c_frag.x[6], c_frag.x[7]);

    // if (bx == 0 && by == 0 && threadIdx.x == 0) {
    //     printf("C:\n");
    //     for (int i = 0; i < 16 * 16; i++) {
    //         printf("%.2f, ", __half2float(C[i]));
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //     }
    // }
#endif
}

template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
void hgemm_v2(const half *A, const half *B, half *C, int M, int N, int K) {
    dim3 grid_dim(ceil_div(N, WMMA_N), ceil_div(M, WMMA_M));
    hgemm_v2_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid_dim, WARP_SIZE>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <int BM, int BN, int BK, int WX, int WY, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void __launch_bounds__(WX *WY *WARP_SIZE)
    hgemm_v3_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M, int N, int K) {

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

warp tiling: (WM = 2 * WMMA_M, WN = 2 * WMMA_N)
                                   Block B
                            +===================+
                            |    |    |    |    |
                            |    |    |    |    |
                            |    |    |    |    |
                            |    |    |    |    |
                            |    |    |    |    |
                            |    |    |    |    |
                            |    |    |    |    |
                            +===================+

                                           WMMA_N
    +===================+   +===================+
    |                   |   | W0 | W1 | W0 | W1 | WMMA_M
    |-------------------|   |----+----+----+----|
    |                   |   | W2 | W3 | W2 | W3 |
    |-------------------|   |----+----+----+----|
    |                   |   | W0 | W1 | W0 | W1 |
    |-------------------|   |----+----+--TILE---|
    |                   |   | W2 | W3 | W2 | W3 |
    +===================+   +===================+
           Block A                 Block C
    */

    static_assert(BM % (WY * WMMA_M) == 0 && BN % (WX * WMMA_N) == 0 && BK % WMMA_K == 0,
                  "unimplemented: invalid template parameters");

    constexpr int NUM_THREADS = WX * WY * WARP_SIZE;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024, "unimplemented: invalid number of threads");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int tid = tx;

    const int wid = tid / WARP_SIZE;
    const int wx = wid % WX;
    const int wy = wid / WX;

    __shared__ half s_A[BM][BK];
    __shared__ half s_B[BK][BN];

    const half *A_block = A + by * BM * K;
    const half *B_block = B + bx * BN;
    half *C_block = C + by * BM * N + bx * BN;

    constexpr int NUM_TILES_M = BM / (WY * WMMA_M);
    constexpr int NUM_TILES_N = BN / (WX * WMMA_N);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frags[NUM_TILES_M][NUM_TILES_N];
#pragma unroll
    for (int m = 0; m < NUM_TILES_M; m++) {
#pragma unroll
        for (int n = 0; n < NUM_TILES_N; n++) {
            wmma::fill_fragment(c_frags[m][n], 0.f);
        }
    }

    for (int k = 0; k < K; k += BK) {

        // load BM * BK tile of A into shared memory
        {
            static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of A");
            static_assert(BK <= NUM_THREADS * 8, "unimplemented: BK is too large");
            constexpr int A_LOAD_TILE_X = BK / 8;
            constexpr int A_LOAD_TILE_Y = NUM_THREADS / A_LOAD_TILE_X;
            const int x = tid % A_LOAD_TILE_X * 8;
#pragma unroll
            for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
                const int y = y_start + tid / A_LOAD_TILE_X;
                cp_async_cg((float4 *)&s_A[y][x], (float4 *)&A_block[y * K + x]);
            }
        }

        // load BK * BN tile of B into shared memory
        {
            static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of B");
            static_assert(BN <= NUM_THREADS * 8, "unimplemented: BN is too large");
            constexpr int B_LOAD_TILE_X = BN / 8;
            constexpr int B_LOAD_TILE_Y = NUM_THREADS / B_LOAD_TILE_X;
            const int x = tid % B_LOAD_TILE_X * 8;
#pragma unroll
            for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
                const int y = y_start + tid / B_LOAD_TILE_X;
                cp_async_cg((float4 *)&s_B[y][x], (float4 *)&B_block[y * N + x]);
            }
        }

        cp_async_commit_group();

        cp_async_wait_group<0>();
        __syncthreads();

#pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frags[NUM_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frags[NUM_TILES_N];

#pragma unroll
            for (int wm = 0; wm < NUM_TILES_M; wm++) {
                wmma::load_matrix_sync(a_frags[wm], &s_A[(wm * WY + wy) * WMMA_M][wk], BK);
            }

#pragma unroll
            for (int wn = 0; wn < NUM_TILES_N; wn++) {
                wmma::load_matrix_sync(b_frags[wn], &s_B[wk][(wn * WX + wx) * WMMA_N], BN);
            }

#pragma unroll
            for (int wm = 0; wm < NUM_TILES_M; wm++) {
#pragma unroll
                for (int wn = 0; wn < NUM_TILES_N; wn++) {
                    wmma::mma_sync(c_frags[wm][wn], a_frags[wm], b_frags[wn], c_frags[wm][wn]);
                }
            }
        }

        __syncthreads();

        A_block += BK;
        B_block += BK * N;
    }

    // store sums to C
#pragma unroll
    for (int m = 0; m < NUM_TILES_M; m++) {
#pragma unroll
        for (int n = 0; n < NUM_TILES_N; n++) {
            auto c_frag_fp16 = fragment_to_half(c_frags[m][n]);
            wmma::store_matrix_sync(&C_block[(m * WY + wy) * WMMA_M * N + (n * WX + wx) * WMMA_N], c_frag_fp16, N,
                                    wmma::mem_row_major);
        }
    }
}

template <int BM = 64, int BN = 64, int BK = 64, int WX = 1, int WY = 1, int WMMA_M = 16, int WMMA_N = 16,
          int WMMA_K = 16>
void hgemm_v3(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(N % BN == 0 && M % BM == 0 && K % BK == 0) << "unimplemented: invalid matrix dimensions";

    dim3 grid_dim(N / BN, M / BM);
    constexpr int block_dim = WX * WY * WARP_SIZE;

    hgemm_v3_kernel<BM, BN, BK, WX, WY, WMMA_M, WMMA_N, WMMA_K><<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <typename T>
__device__ __forceinline__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

void hgemm_cublas(cublasHandle_t handle, const half *A, const half *B, half *C, int M, int N, int K) {
    const half alpha = __float2half(1);
    const half beta = __float2half(0);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
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
    half *A, *B;
    CHECK_CUDA(cudaMallocHost(&A, sizeof(half) * M * K));
    CHECK_CUDA(cudaMallocHost(&B, sizeof(half) * K * N));

    for (int i = 0; i < M * K; i++) {
#ifndef HGEMM_DEBUG
        A[i] = __float2half((uniform() * 2 - 1) / std::sqrt(K));
#else
        A[i] = __float2half(i / 100.f);
#endif
    }

    for (int i = 0; i < K * N; i++) {
#ifndef HGEMM_DEBUG
        B[i] = __float2half((uniform() * 2 - 1) / std::sqrt(K));
#else
        B[i] = __float2half(i / 100.f);
#endif
    }

    half *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

#define MAKE_ITEM(...) {#__VA_ARGS__, __VA_ARGS__}

    std::vector<std::tuple<std::string, std::function<void(const half *, const half *, half *, int, int, int)>>>
        kernels{
            {"cublas", [handle](const half *A, const half *B, half *C, int M, int N,
                                int K) { hgemm_cublas(handle, A, B, C, M, N, K); }},
            MAKE_ITEM(hgemm_v1),
            MAKE_ITEM(hgemm_v2<16, 16, 16>),
            // MAKE_ITEM(hgemm_v2<32, 8, 16>),
            // MAKE_ITEM(hgemm_v2<8, 32, 16>),

            MAKE_ITEM(hgemm_v3<64, 64, 32, 1, 1>),
            MAKE_ITEM(hgemm_v3<64, 64, 32, 1, 2>),
            MAKE_ITEM(hgemm_v3<64, 64, 32, 2, 1>),
            MAKE_ITEM(hgemm_v3<64, 64, 32, 2, 2>),

            MAKE_ITEM(hgemm_v3<64, 64, 64, 1, 1>),
            MAKE_ITEM(hgemm_v3<64, 64, 64, 1, 2>),
            MAKE_ITEM(hgemm_v3<64, 64, 64, 2, 1>),
            MAKE_ITEM(hgemm_v3<64, 64, 64, 2, 2>),

            MAKE_ITEM(hgemm_v3<64, 64, 32, 2, 4>),
            MAKE_ITEM(hgemm_v3<64, 64, 64, 2, 4>),
            MAKE_ITEM(hgemm_v3<64, 64, 64, 4, 4>),
            // MAKE_ITEM(hgemm_v3<64, 64, 64, 4, 8>),
            // MAKE_ITEM(hgemm_v3<64, 64, 32, 8, 4>),
        };

#undef MAKE_ITEM

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    std::vector<PerfRecord> records;

    half *dC_ref;
    CHECK_CUDA(cudaMalloc(&dC_ref, M * N * sizeof(half)));

    hgemm_cublas(handle, dA, dB, dC_ref, M, N, K);

    for (const auto &item : kernels) {
        const auto &name = std::get<0>(item);
        const auto &fn = std::get<1>(item);

        half *dC_opt;
        CHECK_CUDA(cudaMalloc(&dC_opt, M * N * sizeof(half)));
        CHECK_CUDA(cudaMemset(dC_opt, 0, M * N * sizeof(half)));

        fn(dA, dB, dC_opt, M, N, K);

        check_is_close_d(dC_ref, dC_opt, M * N, 1e-2, 1e-2);

        auto perf_fn = [=] { fn(dA, dB, dC_opt, M, N, K); };
        const int warmup = std::max(4096 / M, 1);
        const int active = warmup * 4;
        const float elapsed = timeit(perf_fn, warmup, active);

        const float tflops = (2ull * M * N * K) / 1e12f / elapsed;
        const float bandwidth = (M * K + K * N + M * N) * sizeof(half) / 1e9f / elapsed;

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

    printf("[best] %s vs cublas: %.1f%% (%.3f vs %.3f ms)\n", best_record.name.c_str(),
           cublas_record.elapsed / best_record.elapsed * 100.f, best_record.elapsed * 1e3f,
           cublas_record.elapsed * 1e3f);

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
    // // square matrix
    // {
    //     std::vector<PerfRecord> all_records;
    //     const int dims[]{1024, 2048, 3072, 4096, 6144, 8192};
    //     for (int d : dims) {
    //         auto records = perf(d, d, d);
    //         all_records.insert(all_records.end(), records.begin(), records.end());
    //     }

    //     save_result("output/hgemm_bench_square.csv", all_records);
    // }

    // fixed K to avoid split-K kernels
    {
        std::vector<PerfRecord> all_records;
        constexpr int K = 1024;
        const int dims[]{1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
        for (int d : dims) {
            auto records = perf(d, d, K);
            all_records.insert(all_records.end(), records.begin(), records.end());
        }

        save_result("output/hgemm_bench_fixk.csv", all_records);
    }

    return 0;
}
