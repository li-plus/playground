// Tutorial:
// * https://siboehm.com/articles/22/CUDA-MMM
// * https://zhuanlan.zhihu.com/p/657632577
// MMA:
// * https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma
// Swizzle:
// * https://zhuanlan.zhihu.com/p/27381896431
// * https://zhuanlan.zhihu.com/p/671419093
// CUTLASS docs: https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md

// clang-format off
// profile:
// $ sudo -E $(which ncu) --set full -o profile -f -k hgemm_mma_v2 -s 2 -c 3 ./build/hgemm
// $ sudo -E $(which ncu) --set full -o profile -f --kernel-id ::regex:"ampere_.*|hgemm_mma_v2":3 ./build/hgemm
// clang-format on

#include "common.h"
#include <cstdlib>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <functional>
#include <getopt.h>
#include <mma.h>
#include <sstream>
#include <string>

// #define HGEMM_DEBUG

using namespace nvcuda;

template <typename T>
__device__ __forceinline__ void cp_async_cg(T *dst, const T *src) {
    uint32_t smem_int_ptr = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_int_ptr), "l"(src), "n"(sizeof(T)));
    // cp.async.cg.shared.global.L2::128B
}

__device__ __forceinline__ void cp_async_commit_group() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
    static_assert(n > 0, "n must be positive");
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

template <>
__device__ __forceinline__ void cp_async_wait_group<0>() {
    asm volatile("cp.async.wait_all;\n" ::);
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

    // clang-format off
    /*
wmma layout:
                                                                                                        B frag
                                                                            0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
                                                                        -------------------------------------------------------------------
                                                                        0:  0   4   8   12  16  20  24  28  0   4   8   12  16  20  24  28
                                                                        1:  0   4   8   12  16  20  24  28  0   4   8   12  16  20  24  28
                                                                        2:  1   5   9   13  17  21  25  29  1   5   9   13  17  21  25  29
                                                                        3:  1   5   9   13  17  21  25  29  1   5   9   13  17  21  25  29
                                                                        4:  2   6   10  14  18  22  26  30  2   6   10  14  18  22  26  30
                                                                        5:  2   6   10  14  18  22  26  30  2   6   10  14  18  22  26  30
                                                                        6:  3   7   11  15  19  23  27  31  3   7   11  15  19  23  27  31
                                                                        7:  3   7   11  15  19  23  27  31  3   7   11  15  19  23  27  31
                                                                        8:  0   4   8   12  16  20  24  28  0   4   8   12  16  20  24  28
                                                                        9:  0   4   8   12  16  20  24  28  0   4   8   12  16  20  24  28
                                                                        10: 1   5   9   13  17  21  25  29  1   5   9   13  17  21  25  29
                                                                        11: 1   5   9   13  17  21  25  29  1   5   9   13  17  21  25  29
                                                                        12: 2   6   10  14  18  22  26  30  2   6   10  14  18  22  26  30
                                                                        13: 2   6   10  14  18  22  26  30  2   6   10  14  18  22  26  30
                                                                        14: 3   7   11  15  19  23  27  31  3   7   11  15  19  23  27  31
                                                                        15: 3   7   11  15  19  23  27  31  3   7   11  15  19  23  27  31

    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15           0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
-------------------------------------------------------------------     -------------------------------------------------------------------
0:  0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3       0:  0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3
1:  4   4   5   5   6   6   7   7   4   4   5   5   6   6   7   7       1:  4   4   5   5   6   6   7   7   4   4   5   5   6   6   7   7
2:  8   8   9   9   10  10  11  11  8   8   9   9   10  10  11  11      2:  8   8   9   9   10  10  11  11  8   8   9   9   10  10  11  11
3:  12  12  13  13  14  14  15  15  12  12  13  13  14  14  15  15      3:  12  12  13  13  14  14  15  15  12  12  13  13  14  14  15  15
4:  16  16  17  17  18  18  19  19  16  16  17  17  18  18  19  19      4:  16  16  17  17  18  18  19  19  16  16  17  17  18  18  19  19
5:  20  20  21  21  22  22  23  23  20  20  21  21  22  22  23  23      5:  20  20  21  21  22  22  23  23  20  20  21  21  22  22  23  23
6:  24  24  25  25  26  26  27  27  24  24  25  25  26  26  27  27      6:  24  24  25  25  26  26  27  27  24  24  25  25  26  26  27  27
7:  28  28  29  29  30  30  31  31  28  28  29  29  30  30  31  31      7:  28  28  29  29  30  30  31  31  28  28  29  29  30  30  31  31
8:  0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3       8:  0   0   1   1   2   2   3   3   0   0   1   1   2   2   3   3
9:  4   4   5   5   6   6   7   7   4   4   5   5   6   6   7   7       9:  4   4   5   5   6   6   7   7   4   4   5   5   6   6   7   7
10: 8   8   9   9   10  10  11  11  8   8   9   9   10  10  11  11      10: 8   8   9   9   10  10  11  11  8   8   9   9   10  10  11  11
11: 12  12  13  13  14  14  15  15  12  12  13  13  14  14  15  15      11: 12  12  13  13  14  14  15  15  12  12  13  13  14  14  15  15
12: 16  16  17  17  18  18  19  19  16  16  17  17  18  18  19  19      12: 16  16  17  17  18  18  19  19  16  16  17  17  18  18  19  19
13: 20  20  21  21  22  22  23  23  20  20  21  21  22  22  23  23      13: 20  20  21  21  22  22  23  23  20  20  21  21  22  22  23  23
14: 24  24  25  25  26  26  27  27  24  24  25  25  26  26  27  27      14: 24  24  25  25  26  26  27  27  24  24  25  25  26  26  27  27
15: 28  28  29  29  30  30  31  31  28  28  29  29  30  30  31  31      15: 28  28  29  29  30  30  31  31  28  28  29  29  30  30  31  31
                        A frag                                                              C frag (same layout as A)
    */
    // clang-format on

#ifdef HGEMM_DEBUG
    printf("block (%d, %d) thread (%d, %d) a_frag(%d): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] b_frag(%d): "
           "[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] c_frag(%d): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, "
           "%.2f]\n",
           bx, by, threadIdx.x, threadIdx.y, a_frag.num_storage_elements, __half2float(a_frag.x[0]),
           __half2float(a_frag.x[1]), __half2float(a_frag.x[2]), __half2float(a_frag.x[3]), __half2float(a_frag.x[4]),
           __half2float(a_frag.x[5]), __half2float(a_frag.x[6]), __half2float(a_frag.x[7]), b_frag.num_storage_elements,
           __half2float(b_frag.x[0]), __half2float(b_frag.x[1]), __half2float(b_frag.x[2]), __half2float(b_frag.x[3]),
           __half2float(b_frag.x[4]), __half2float(b_frag.x[5]), __half2float(b_frag.x[6]), __half2float(b_frag.x[7]),
           c_frag.num_storage_elements, c_frag.x[0], c_frag.x[1], c_frag.x[2], c_frag.x[3], c_frag.x[4], c_frag.x[5],
           c_frag.x[6], c_frag.x[7]);
#endif
}

template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
void hgemm_v2(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / WMMA_N, M / WMMA_M);
    hgemm_v2_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid_dim, WARP_SIZE>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <int BM, int BN, int BK, int WX, int WY, int WMMA_M, int WMMA_N, int WMMA_K, int PAD_A, int PAD_B>
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

    // pad to reduce wmma::load_matrix_sync bank conflict
    __shared__ half s_A[BM][BK + PAD_A];
    __shared__ half s_B[BK][BN + PAD_B];

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

    // utils for global -> shared
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of A");
    static_assert(BK <= NUM_THREADS * 8, "unimplemented: BK is too large");
    constexpr int A_LOAD_TILE_Y = NUM_THREADS * 8 / BK;
    const int A_x = tid * 8 % BK;
    const int A_y = tid * 8 / BK;

    static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of B");
    static_assert(BN <= NUM_THREADS * 8, "unimplemented: BN is too large");
    constexpr int B_LOAD_TILE_Y = NUM_THREADS * 8 / BN;
    const int B_x = tid * 8 % BN;
    const int B_y = tid * 8 / BN;

    for (int k = 0; k < K; k += BK) {

        // load BM * BK tile of A into shared memory
#pragma unroll
        for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
            const int y = y_start + A_y;
            cp_async_cg((float4 *)&s_A[y][A_x], (float4 *)&A_block[y * K + A_x]);
        }

        // load BK * BN tile of B into shared memory
#pragma unroll
        for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
            const int y = y_start + B_y;
            cp_async_cg((float4 *)&s_B[y][B_x], (float4 *)&B_block[y * N + B_x]);
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
                wmma::load_matrix_sync(a_frags[wm], &s_A[(wm * WY + wy) * WMMA_M][wk], BK + PAD_A);
            }

#pragma unroll
            for (int wn = 0; wn < NUM_TILES_N; wn++) {
                wmma::load_matrix_sync(b_frags[wn], &s_B[wk][(wn * WX + wx) * WMMA_N], BN + PAD_B);
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
          int WMMA_K = 16, int PAD_A = 16, int PAD_B = 16>
void hgemm_v3(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / BN, M / BM);
    constexpr int block_dim = WX * WY * WARP_SIZE;
    hgemm_v3_kernel<BM, BN, BK, WX, WY, WMMA_M, WMMA_N, WMMA_K, PAD_A, PAD_B>
        <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

// multi-stage
template <int BM, int BN, int BK, int WX, int WY, int STAGES, int WMMA_M, int WMMA_N, int WMMA_K, int PAD_A, int PAD_B>
__global__ void __launch_bounds__(WX *WY *WARP_SIZE)
    hgemm_v4_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M, int N, int K) {

    static_assert(STAGES > 1, "unimplemented: STAGES must be greater than 1");
    static_assert(BM % (WY * WMMA_M) == 0 && BN % (WX * WMMA_N) == 0 && BK % WMMA_K == 0,
                  "unimplemented: invalid template parameters");

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-description
    static_assert(PAD_A % 16 == 0 && PAD_B % 16 == 0, "A&B address must be 256-bit aligned");

    constexpr int NUM_THREADS = WX * WY * WARP_SIZE;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024, "unimplemented: invalid number of threads");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int tid = tx;

    const int wid = tid / WARP_SIZE;
    const int wx = wid % WX;
    const int wy = wid / WX;

    __shared__ half s_A[STAGES][BM][BK + PAD_A];
    __shared__ half s_B[STAGES][BK][BN + PAD_B];

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

    // ===== fetch block =====
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of A");
    static_assert(BK <= NUM_THREADS * 8, "unimplemented: BK is too large");
    constexpr int A_LOAD_TILE_Y = NUM_THREADS * 8 / BK;
    const int A_x = tid * 8 % BK;
    const int A_y = tid * 8 / BK;

    static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of B");
    static_assert(BN <= NUM_THREADS * 8, "unimplemented: BN is too large");
    constexpr int B_LOAD_TILE_Y = NUM_THREADS * 8 / BN;
    const int B_x = tid * 8 % BN;
    const int B_y = tid * 8 / BN;

    auto fetch_block = [&](int i) {
        const int stage = i % STAGES;

        // load BM * BK tile of A into shared memory
        const half *A_block = A + by * BM * K + i * BK;
#pragma unroll
        for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
            const int y = y_start + A_y;
            cp_async_cg((float4 *)&s_A[stage][y][A_x], (float4 *)&A_block[y * K + A_x]);
        }

        // load BK * BN tile of B into shared memory
        const half *B_block = B + bx * BN + i * BK * N;
#pragma unroll
        for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
            const int y = y_start + B_y;
            cp_async_cg((float4 *)&s_B[stage][y][B_x], (float4 *)&B_block[y * N + B_x]);
        }
    };

    // ===== mma =====
    auto mma_compute = [&](int i) {
        const int stage = i % STAGES;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frags[NUM_TILES_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frags[NUM_TILES_N];

#pragma unroll
        for (int wk = 0; wk < BK; wk += WMMA_K) {
#pragma unroll
            for (int wm = 0; wm < NUM_TILES_M; wm++) {
                wmma::load_matrix_sync(a_frags[wm], &s_A[stage][(wm * WY + wy) * WMMA_M][wk], BK + PAD_A);
            }

#pragma unroll
            for (int wn = 0; wn < NUM_TILES_N; wn++) {
                wmma::load_matrix_sync(b_frags[wn], &s_B[stage][wk][(wn * WX + wx) * WMMA_N], BN + PAD_B);
            }

#pragma unroll
            for (int wm = 0; wm < NUM_TILES_M; wm++) {
#pragma unroll
                for (int wn = 0; wn < NUM_TILES_N; wn++) {
                    wmma::mma_sync(c_frags[wm][wn], a_frags[wm], b_frags[wn], c_frags[wm][wn]);
                }
            }
        }
    };

#pragma unroll
    for (int i = 0; i < STAGES - 1; i++) {
        fetch_block(i);
        cp_async_commit_group();
    }

    for (int i = STAGES - 1; i < K / BK; i++) {
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();

        fetch_block(i);
        cp_async_commit_group();

        mma_compute(i - (STAGES - 1));
    }

    cp_async_wait_group<0>();
    __syncthreads();

#pragma unroll
    for (int i = -(STAGES - 1); i < 0; i++) {
        mma_compute(K / BK + i);
    }

    // store sums to C
    half *C_block = C + by * BM * N + bx * BN;
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

template <int BM = 64, int BN = 64, int BK = 64, int WX = 1, int WY = 1, int STAGES = 2, int WMMA_M = 16,
          int WMMA_N = 16, int WMMA_K = 16, int PAD_A = 16, int PAD_B = 16>
void hgemm_v4(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / BN, M / BM);
    constexpr int block_dim = WX * WY * WARP_SIZE;
    hgemm_v4_kernel<BM, BN, BK, WX, WY, STAGES, WMMA_M, WMMA_N, WMMA_K, PAD_A, PAD_B>
        <<<grid_dim, block_dim>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

__device__ __forceinline__ void ldmatrix(uint &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(dst) : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix(uint2 &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(dst.x), "=r"(dst.y)
                 : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix(uint4 &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
                 : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_trans(uint &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n" : "=r"(dst) : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_trans(uint2 &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(dst.x), "=r"(dst.y)
                 : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_trans(uint4 &dst, half *src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(src);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
                 : "r"(smem_ptr));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
__device__ __forceinline__ void mma_m16n8k16(uint4 &d, uint4 a, uint2 b, uint4 c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
}

__device__ __forceinline__ void mma_m16n8k16(uint2 &d, uint4 a, uint2 b, uint2 c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                 : "=r"(d.x), "=r"(d.y)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(c.x), "r"(c.y));
}

template <typename T>
__device__ __forceinline__ constexpr T ce_max(const T &a, const T &b) {
    return a > b ? a : b;
}

// mma kernel
template <int MMA_M, int MMA_N, int MMA_K>
__global__ void hgemm_mma_v1_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M,
                                    int N, int K) {
    static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16, "unimplemented: only support m16n8k16");

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int tid = tx;

    __shared__ half s_A[MMA_M][MMA_K]; // 16 * 16
    __shared__ half s_B[MMA_K][MMA_N]; // 16 * 8

    const half *A_block = A + by * MMA_M * K;
    const half *B_block = B + bx * MMA_N;
    half *C_block = C + by * MMA_M * N + bx * MMA_N;

    half a_frag[8];
    half b_frag[4];
    float c_frag[4]{};

    for (int k = 0; k < K; k += MMA_K) {
        const int A_x = tid * 8 % MMA_K;
        const int A_y = tid * 8 / MMA_K;
        *(float4 *)&s_A[A_y][A_x] = *(float4 *)&A_block[A_y * K + A_x];

        const int B_x = tid * 4 % MMA_N;
        const int B_y = tid * 4 / MMA_N;
        *(float2 *)&s_B[B_y][B_x] = *(float2 *)&B_block[B_y * N + B_x];

        __syncthreads();

        ldmatrix(*(uint4 *)a_frag, &s_A[tid % MMA_M][(tid / MMA_M) * 8]);
        ldmatrix_trans(*(uint2 *)b_frag, &s_B[tid % MMA_K][0]);

        mma_m16n8k16(*(uint4 *)c_frag, *(uint4 *)a_frag, *(uint2 *)b_frag, *(uint4 *)c_frag);

        A_block += MMA_K;
        B_block += MMA_K * N;
    }

    const int C_x = tid * 2 % MMA_N;
    const int C_y = tid * 2 / MMA_N;

#pragma unroll
    for (int i = 0; i < 2; i++) {
        *(half2 *)&C_block[C_y * N + C_x] = __float22half2_rn(*(float2 *)&c_frag[i * 2]);
        C_block += (WARP_SIZE * 2 / MMA_N) * N;
    }

#ifdef HGEMM_DEBUG
    printf("block (%d, %d) thread (%d, %d) a_frag: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f] b_frag: [%.2f, "
           "%.2f, %.2f, %.2f] c_frag: [%.2f, %.2f, %.2f, %.2f]\n",
           bx, by, threadIdx.x, threadIdx.y, __half2float(a_frag[0]), __half2float(a_frag[1]), __half2float(a_frag[2]),
           __half2float(a_frag[3]), __half2float(a_frag[4]), __half2float(a_frag[5]), __half2float(a_frag[6]),
           __half2float(a_frag[7]), __half2float(b_frag[0]), __half2float(b_frag[1]), __half2float(b_frag[2]),
           __half2float(b_frag[3]), c_frag[0], c_frag[1], c_frag[2], c_frag[3]);
#endif
}

template <int MMA_M = 16, int MMA_N = 8, int MMA_K = 16>
void hgemm_mma_v1(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % MMA_M == 0 && N % MMA_N == 0 && K % MMA_K == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / MMA_N, M / MMA_M);
    hgemm_mma_v1_kernel<MMA_M, MMA_N, MMA_K><<<grid_dim, WARP_SIZE>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

template <int _2_M, int _2_S>
__device__ __forceinline__ constexpr int swizzle_permute(int index) {
    const int group_index = index / _2_M;
    const int group_y = group_index / _2_S;
    const int group_x = (group_index ^ group_y) % _2_S;
    return (group_y * _2_S + group_x) * _2_M + index % _2_M;
}

// mma kernel
template <int BM, int BN, int BK, int WX, int WY, int STAGES, bool SWIZZLE, int BLOCK_SWIZZLE_SIZE, int MMA_M,
          int MMA_N, int MMA_K>
__global__ void __launch_bounds__(WX *WY *WARP_SIZE)
    hgemm_mma_v2_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M, int N,
                        int K) {
    static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16, "unimplemented: only support m16n8k16");

    static_assert(BM % (WY * MMA_M) == 0 && BN % (WX * MMA_N) == 0 && BK % MMA_K == 0,
                  "unimplemented: invalid template parameters");

    constexpr int NUM_THREADS = WX * WY * WARP_SIZE;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024, "unimplemented: invalid number of threads");

    int _bx = blockIdx.x;
    int _by = blockIdx.y;
    if constexpr (BLOCK_SWIZZLE_SIZE > 1) {
        const int gx = gridDim.x;
        const int bid = _by * gx + _bx;
        _bx = bid / BLOCK_SWIZZLE_SIZE % gx;
        _by = bid % BLOCK_SWIZZLE_SIZE + bid / gx / BLOCK_SWIZZLE_SIZE * BLOCK_SWIZZLE_SIZE;
    }
    const int bx = _bx;
    const int by = _by;

    const int tx = threadIdx.x;
    const int tid = tx;

    const int wid = tid / WARP_SIZE; // warp_id
    const int wx = wid % WX;
    const int wy = wid / WX;

    const int lid = tid % WARP_SIZE; // lane_id

    extern __shared__ half shm[];

    auto s_A = (half(*)[BM][BK])shm;                      // [STAGES][BM][BK]
    auto s_B = (half(*)[BK][BN])(shm + STAGES * BM * BK); // [STAGES][BK][BN];

    constexpr int NUM_MMA_M = BM / (WY * MMA_M);
    constexpr int NUM_MMA_N = BN / (WX * MMA_N);

    uint2 c_frags[NUM_MMA_M][NUM_MMA_N]{};

    // swizzle
    constexpr int SWIZZLE_2_M = 8;                              // 2^3=8 elements as a unit
    constexpr int SWIZZLE_A_2_S = ce_max(64, BK) / SWIZZLE_2_M; // at least 2^3=8 elements per row
    constexpr int SWIZZLE_B_2_S = ce_max(64, BN) / SWIZZLE_2_M; // at least 2^3=8 elements per row

    // ===== fetch block =====
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of A");
    static_assert(BK <= NUM_THREADS * 8, "unimplemented: BK is too large");
    constexpr int A_LOAD_TILE_Y = NUM_THREADS * 8 / BK;
    const int A_x = tid * 8 % BK;
    const int A_y = tid * 8 / BK;

    static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of B");
    static_assert(BN <= NUM_THREADS * 8, "unimplemented: BN is too large");
    constexpr int B_LOAD_TILE_Y = NUM_THREADS * 8 / BN;
    const int B_x = tid * 8 % BN;
    const int B_y = tid * 8 / BN;

    auto fetch_block = [&](int i) {
        const int stage = i % STAGES;

        // load BM * BK tile of A into shared memory
        const half *A_block = A + by * BM * K + i * BK;
#pragma unroll
        for (int y_start = 0; y_start < BM; y_start += A_LOAD_TILE_Y) {
            const int y = y_start + A_y;
            if constexpr (SWIZZLE) {
                const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_A_2_S>(y * BK + A_x);
                cp_async_cg((float4 *)&s_A[stage][0][offset], (float4 *)&A_block[y * K + A_x]);
            } else {
                cp_async_cg((float4 *)&s_A[stage][y][A_x], (float4 *)&A_block[y * K + A_x]);
            }
        }

        // load BK * BN tile of B into shared memory
        const half *B_block = B + bx * BN + i * BK * N;
#pragma unroll
        for (int y_start = 0; y_start < BK; y_start += B_LOAD_TILE_Y) {
            const int y = y_start + B_y;
            if constexpr (SWIZZLE) {
                const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_B_2_S>(y * BN + B_x);
                cp_async_cg((float4 *)&s_B[stage][0][offset], (float4 *)&B_block[y * N + B_x]);
            } else {
                cp_async_cg((float4 *)&s_B[stage][y][B_x], (float4 *)&B_block[y * N + B_x]);
            }
        }
    };

    // ===== mma =====
    auto mma_compute = [&](int i) {
        const int stage = i % STAGES;

#pragma unroll
        for (int wk = 0; wk < BK; wk += MMA_K) {
            uint4 a_frags[NUM_MMA_M];
            uint2 b_frags[NUM_MMA_N];

            auto load_a_frags = [&] {
#pragma unroll
                for (int wm = 0; wm < NUM_MMA_M; wm++) {
                    const int row_offset = (wm * WY + wy) * MMA_M + lid % MMA_M;
                    const int col_offset = wk + lid / MMA_M * 8;
                    if constexpr (SWIZZLE) {
                        const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_A_2_S>(row_offset * BK + col_offset);
                        // if (bx == 0 && by == 0 && wid == 1 && i == 0 && wk == 0 && wm == 0) {
                        //     printf("[tid %d] ldmatrix A (%d, %d)\n", tid, offset / 8 / 8, offset / 8 % 8);
                        // }
                        ldmatrix(a_frags[wm], &s_A[stage][0][offset]);
                    } else {
                        ldmatrix(a_frags[wm], &s_A[stage][row_offset][col_offset]);
                    }
                }
            };

            auto load_b_frags = [&] {
#pragma unroll
                for (int wn = 0; wn < NUM_MMA_N; wn++) {
                    const int row_offset = wk + lid % MMA_K;
                    const int col_offset = (wn * WX + wx) * MMA_N;
                    if constexpr (SWIZZLE) {
                        const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_B_2_S>(row_offset * BN + col_offset);
                        // if (bx == 0 && by == 0 && wid == 1 && i == 0 && wk == 0 && wn == 0) {
                        //     printf("[tid %d] ldmatrix B (%d, %d)\n", tid, offset / 8 / 8, offset / 8 % 8);
                        // }
                        ldmatrix_trans(b_frags[wn], &s_B[stage][0][offset]);
                    } else {
                        ldmatrix_trans(b_frags[wn], &s_B[stage][row_offset][col_offset]);
                    }
                }
            };

            if (wid % 2 == 1) {
                load_a_frags();
                load_b_frags();
            } else {
                load_b_frags();
                load_a_frags();
            }

#pragma unroll
            for (int wm = 0; wm < NUM_MMA_M; wm++) {
#pragma unroll
                for (int wn = 0; wn < NUM_MMA_N; wn++) {
                    mma_m16n8k16(c_frags[wm][wn], a_frags[wm], b_frags[wn], c_frags[wm][wn]);
                }
            }
        }
    };

#pragma unroll
    for (int i = 0; i < STAGES - 1; i++) {
        fetch_block(i);
        cp_async_commit_group();
    }

    for (int i = STAGES - 1; i < K / BK; i++) {
        if constexpr (STAGES > 1) {
            cp_async_wait_group<STAGES - 2>();
        }
        __syncthreads();

        fetch_block(i);
        cp_async_commit_group();

        if constexpr (STAGES == 1) {
            cp_async_wait_group<0>();
            __syncthreads();
        }

        mma_compute(i - (STAGES - 1));
    }

    if constexpr (STAGES > 1) {
        cp_async_wait_group<0>();
        __syncthreads();
    }

#pragma unroll
    for (int i = -(STAGES - 1); i < 0; i++) {
        mma_compute(K / BK + i);
    }

    // store sums to C
    half *C_block = C + by * BM * N + bx * BN;
    const int C_x = lid * 2 % MMA_N;
    const int C_y = lid * 2 / MMA_N;
#pragma unroll
    for (int m = 0; m < NUM_MMA_M; m++) {
#pragma unroll
        for (int n = 0; n < NUM_MMA_N; n++) {
            half *C_tile = C_block + (m * WY + wy) * MMA_M * N + (n * WX + wx) * MMA_N;
#pragma unroll
            for (int i = 0; i < 2; i++) {
                *(half2 *)&C_tile[(C_y + i * 8) * N + C_x] = ((half2 *)&c_frags[m][n])[i];
            }
        }
    }
}

template <int BM, int BN, int BK, int WX, int WY, int STAGES, bool SWIZZLE = false, int BLOCK_SWIZZLE_SIZE = 1,
          int MMA_M = 16, int MMA_N = 8, int MMA_K = 16>
void hgemm_mma_v2(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / BN, M / BM);
    constexpr int block_dim = WX * WY * WARP_SIZE;
    constexpr int shared_mem_size = STAGES * (BM + BN) * BK * sizeof(half);
    CHECK_CUDA(cudaFuncSetAttribute(
        hgemm_mma_v2_kernel<BM, BN, BK, WX, WY, STAGES, SWIZZLE, BLOCK_SWIZZLE_SIZE, MMA_M, MMA_N, MMA_K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    hgemm_mma_v2_kernel<BM, BN, BK, WX, WY, STAGES, SWIZZLE, BLOCK_SWIZZLE_SIZE, MMA_M, MMA_N, MMA_K>
        <<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

// aggressive optimization for peak performance
template <int BM, int BN, int BK, int WX, int WY, int STAGES, int BLOCK_SWIZZLE_SIZE, bool FEAT, int MMA_M, int MMA_N,
          int MMA_K>
__global__ void __launch_bounds__(WX *WY *WARP_SIZE)
    hgemm_mma_v3_kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, int M, int N,
                        int K) {
    static_assert(MMA_M == 16 && MMA_N == 8 && MMA_K == 16, "unimplemented: only support m16n8k16");

    static_assert(BM % (WY * MMA_M) == 0 && BN % (WX * MMA_N * 2) == 0 && BK % MMA_K == 0,
                  "unimplemented: invalid template parameters");

    constexpr int NUM_THREADS = WX * WY * WARP_SIZE;
    static_assert(32 <= NUM_THREADS && NUM_THREADS <= 1024, "unimplemented: invalid number of threads");
    static_assert(STAGES > 1, "unimplemented: STAGES must be greater than 1");

    int _bx = blockIdx.x;
    int _by = blockIdx.y;
    if (BLOCK_SWIZZLE_SIZE > 1) {
        const int gx = gridDim.x;
        const int bid = _by * gx + _bx;
        _bx = bid / BLOCK_SWIZZLE_SIZE % gx;
        _by = bid % BLOCK_SWIZZLE_SIZE + bid / gx / BLOCK_SWIZZLE_SIZE * BLOCK_SWIZZLE_SIZE;
    }
    const int bx = _bx;
    const int by = _by;

    const int tx = threadIdx.x;
    const int tid = tx;

    const int lid = tid % WARP_SIZE; // lane_id
    const int wid = tid / WARP_SIZE; // warp_id
    const int wx = wid % WX;
    const int wy = wid / WX;

    extern __shared__ half smem[];

    auto s_A = (half(*)[BM][BK])smem;                      // [STAGES][BM][BK]
    auto s_B = (half(*)[BK][BN])(smem + STAGES * BM * BK); // [STAGES][BK][BN]

    constexpr int NUM_MMA_M = BM / (WY * MMA_M);
    constexpr int NUM_MMA_N = BN / (WX * MMA_N * 2);
    constexpr int NUM_MMA_K = BK / MMA_K;

    uint4 c_frags[NUM_MMA_M][NUM_MMA_N]{};

    // swizzle
    constexpr int SWIZZLE_2_M = 8;                              // 2^3=8 elements as a unit
    constexpr int SWIZZLE_A_2_S = ce_max(64, BK) / SWIZZLE_2_M; // at least 2^3=8 elements per row
    constexpr int SWIZZLE_B_2_S = ce_max(64, BN) / SWIZZLE_2_M; // at least 2^3=8 elements per row

    // ===== fetch block =====
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of A");
    static_assert(BK <= NUM_THREADS * 8, "unimplemented: BK is too large");
    constexpr int A_LOAD_TILE_Y = NUM_THREADS * 8 / BK;
    constexpr int A_NUM_LDGSTS = BM / A_LOAD_TILE_Y;
    const int A_x = tid * 8 % BK;
    const int A_y = tid * 8 / BK;

    static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "unimplemented: corrupted load of B");
    static_assert(BN <= NUM_THREADS * 8, "unimplemented: BN is too large");
    constexpr int B_LOAD_TILE_Y = NUM_THREADS * 8 / BN;
    constexpr int B_NUM_LDGSTS = BK / B_LOAD_TILE_Y;
    const int B_x = tid * 8 % BN;
    const int B_y = tid * 8 / BN;

    auto fetch_A_block = [&](int k) {
        const int stage = k % STAGES;
        const half *A_block = A + by * BM * K + k * BK;
#pragma unroll
        for (int tile_i = 0; tile_i < A_NUM_LDGSTS; tile_i++) {
            const int y = tile_i * A_LOAD_TILE_Y + A_y;
            const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_A_2_S>(y * BK + A_x);
            cp_async_cg((float4 *)&s_A[stage][0][offset], (float4 *)&A_block[y * K + A_x]);
        }
    };
    auto fetch_B_block = [&](int k) {
        const int stage = k % STAGES;
        const half *B_block = B + bx * BN + k * BK * N;
#pragma unroll
        for (int tile_i = 0; tile_i < B_NUM_LDGSTS; tile_i++) {
            const int y = tile_i * B_LOAD_TILE_Y + B_y;
            const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_B_2_S>(y * BN + B_x);
            cp_async_cg((float4 *)&s_B[stage][0][offset], (float4 *)&B_block[y * N + B_x]);
        }
    };

    uint4 a_frags[2][NUM_MMA_M];
    uint4 b_frags[2][NUM_MMA_N];

    auto load_a_frags = [&](int k, int mma_k) {
        const int stage = k % STAGES;
        const int stage_ldsm = mma_k % 2;
        const int col_offset = mma_k * MMA_K + lid / MMA_M * 8;
#pragma unroll
        for (int mma_m = 0; mma_m < NUM_MMA_M; mma_m++) {
            const int row_offset = (mma_m * WY + wy) * MMA_M + lid % MMA_M;
            const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_A_2_S>(row_offset * BK + col_offset);
            ldmatrix(a_frags[stage_ldsm][mma_m], &s_A[stage][0][offset]);
        }
    };

    auto load_b_frags = [&](int k, int mma_k) {
        const int stage = k % STAGES;
        const int stage_ldsm = mma_k % 2;
        const int row_offset = mma_k * MMA_K + lid % MMA_K;
#pragma unroll
        for (int mma_n = 0; mma_n < NUM_MMA_N; mma_n++) {
            const int col_offset = (mma_n * WX + wx) * (2 * MMA_N) + lid / MMA_K * 8;
            const int offset = swizzle_permute<SWIZZLE_2_M, SWIZZLE_B_2_S>(row_offset * BN + col_offset);
            ldmatrix_trans(b_frags[stage_ldsm][mma_n], &s_B[stage][0][offset]);
        }
    };

    auto mma_frags = [&](int mma_k) {
        const int stage_ldsm = mma_k % 2;

#pragma unroll
        for (int mma_m = 0; mma_m < NUM_MMA_M; mma_m++) {
#pragma unroll
            for (int mma_n = 0; mma_n < NUM_MMA_N; mma_n++) {
                mma_m16n8k16(((uint2 *)&c_frags[mma_m][mma_n])[0], a_frags[stage_ldsm][mma_m],
                             ((uint2 *)&b_frags[stage_ldsm][mma_n])[0], ((uint2 *)&c_frags[mma_m][mma_n])[0]);
                mma_m16n8k16(((uint2 *)&c_frags[mma_m][mma_n])[1], a_frags[stage_ldsm][mma_m],
                             ((uint2 *)&b_frags[stage_ldsm][mma_n])[1], ((uint2 *)&c_frags[mma_m][mma_n])[1]);
            }
        }
    };

#pragma unroll
    for (int k = 0; k < STAGES - 1; k++) {
        // LDGSTS
        fetch_A_block(k);
        fetch_B_block(k);
        cp_async_commit_group();
    }

    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // LDSM
    load_a_frags(0, 0);
    load_b_frags(0, 0);

    // main loop over k
    for (int k = STAGES - 1; k < K / BK; k++) {

        // LDGSTS
        fetch_A_block(k);
        fetch_B_block(k);
        cp_async_commit_group();

#pragma unroll
        for (int mma_k = 0; mma_k < NUM_MMA_K; mma_k++) {
            // LDSM
            const int ldsm_mma_k = (mma_k + 1) % NUM_MMA_K;
            const int ldsm_k = k - (STAGES - 1) + (mma_k + 1) / NUM_MMA_K;
            if (mma_k + 1 == NUM_MMA_K) {
                cp_async_wait_group<STAGES - 2>();
                __syncthreads();
            }
            load_a_frags(ldsm_k, ldsm_mma_k);
            load_b_frags(ldsm_k, ldsm_mma_k);

            // MMA
            mma_frags(mma_k);
        }
    }

    cp_async_wait_group<0>();
    __syncthreads();

#pragma unroll
    for (int i = -(STAGES - 1); i < 0; i++) {
        const int k = K / BK + i;

#pragma unroll
        for (int mma_k = 0; mma_k < NUM_MMA_K; mma_k++) {
            // LDSM
            const int ldsm_mma_k = (mma_k + 1) % NUM_MMA_K;
            const int ldsm_k = k + (mma_k + 1) / NUM_MMA_K;
            if (ldsm_k < K / BK) {
                load_a_frags(ldsm_k, ldsm_mma_k);
                load_b_frags(ldsm_k, ldsm_mma_k);
            }

            // MMA
            mma_frags(mma_k);
        }
    }

    // store sums to C
    // store to shared memory first is faster than directly to global memory
    const int C_stsm_x = lid * 2 % MMA_N;
    const int C_stsm_y = lid * 2 / MMA_N;

    static_assert(STAGES * (BM + BN) * BK >= BM * BN, "unimplemented: not enough shared memory");

    constexpr int SWIZZLE_C_2_S = ce_max(64, BN) / SWIZZLE_2_M;
    auto swizzle_fn = swizzle_permute<SWIZZLE_2_M, SWIZZLE_C_2_S>;

    __syncthreads();

#pragma unroll
    for (int m = 0; m < NUM_MMA_M; m++) {
        // STSM
#pragma unroll
        for (int n = 0; n < NUM_MMA_N; n++) {
            const int stsm_y = (m * WY + wy) * MMA_M + C_stsm_y;
            const int stsm_x = (n * WX + wx) * (2 * MMA_N) + C_stsm_x;
            *(uint *)&smem[swizzle_fn(stsm_y * BN + stsm_x)] = c_frags[m][n].x;
            *(uint *)&smem[swizzle_fn(stsm_y * BN + stsm_x + 8)] = c_frags[m][n].z;
            *(uint *)&smem[swizzle_fn((stsm_y + 8) * BN + stsm_x)] = c_frags[m][n].y;
            *(uint *)&smem[swizzle_fn((stsm_y + 8) * BN + stsm_x + 8)] = c_frags[m][n].w;
        }

        __syncthreads();

        // STG
        half *C_block = C + by * BM * N + bx * BN;

        constexpr int C_STORE_TILE_Y = NUM_THREADS * 8 / BN;
        const int C_stg_x = tid * 8 % BN;
        const int C_stg_y = tid * 8 / BN;

#pragma unroll
        for (int tile_i = m * NUM_MMA_N; tile_i < (m + 1) * NUM_MMA_N; tile_i++) {
            const int stg_y = tile_i * C_STORE_TILE_Y + C_stg_y;
            const int offset = swizzle_fn(stg_y * BN + C_stg_x);
            *(float4 *)&C_block[stg_y * N + C_stg_x] = *(float4 *)&smem[offset];
        }
    }
}

template <int BM, int BN, int BK, int WX, int WY, int STAGES, int BLOCK_SWIZZLE_SIZE = 1, bool FEAT = false,
          int MMA_M = 16, int MMA_N = 8, int MMA_K = 16>
void hgemm_mma_v3(const half *A, const half *B, half *C, int M, int N, int K) {
    CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0) << "unimplemented: invalid matrix dimensions";
    dim3 grid_dim(N / BN, M / BM);
    constexpr int block_dim = WX * WY * WARP_SIZE;
    constexpr int shared_mem_size = STAGES * (BM + BN) * BK * sizeof(half);
    auto kernel_fn = hgemm_mma_v3_kernel<BM, BN, BK, WX, WY, STAGES, BLOCK_SWIZZLE_SIZE, FEAT, MMA_M, MMA_N, MMA_K>;
    CHECK_CUDA(cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    kernel_fn<<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}

void hgemm_cublas(cublasHandle_t handle, const half *A, const half *B, half *C, int M, int N, int K) {
    const half alpha = __float2half(1);
    const half beta = __float2half(0);
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

void hgemm(const half *A, const half *B, half *C, int M, int N, int K);

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

__global__ void uniform_kernel(half *buffer, int N, float start, float end) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(42, idx, 0, &state);
    const half2 weight = __float2half2_rn(end - start);
    const half2 bias = __float2half2_rn(start);

    for (int i = 8 * idx; i < N; i += 8 * blockDim.x * gridDim.x) {
        half2 x[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            x[j] = __float22half2_rn({curand_uniform(&state), curand_uniform(&state)});
            x[j] = __hfma2(x[j], weight, bias);
        }
        *(float4 *)&buffer[i] = *(float4 *)x;
    }
}

void uniform_cuda(half *buffer, int N, float start, float end) {
    CHECK(N % 8 == 0) << "N must be multiple of 8";
    constexpr int block_size = 64;
    const int grid_size = std::min(32768, ceil_div(N / 8, block_size));
    uniform_kernel<<<grid_size, block_size>>>(buffer, N, start, end);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void is_close_kernel(const half *A, const half *B, int N, float atol, float rtol, bool *failure) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx * 8; i < N; i += blockDim.x * gridDim.x * 8) {
        half2 ah[4], bh[4];
        *(float4 *)ah = *(float4 *)&A[i];
        *(float4 *)bh = *(float4 *)&B[i];

        float af[8], bf[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            ((float2 *)af)[j] = __half22float2(ah[j]);
            ((float2 *)bf)[j] = __half22float2(bh[j]);
        }

#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float a = af[j];
            const float b = bf[j];
            const float diff = fabs(a - b);
            const float tol = atol + rtol * fabs(b);
            if (diff > tol) {
                *failure = true;
                // printf("[idx %d] a=%f b=%f diff=%f tol=%f\n", i, a, b, diff, tol);
                // asm("trap;");
            }
        }
    }
}

bool is_close_cuda(const half *A, const half *B, int N, float atol, float rtol) {
    CHECK(N % 8 == 0) << "N must be multiple of 8";

    bool *d_failure;
    CHECK_CUDA(cudaMalloc(&d_failure, sizeof(bool)));
    CHECK_CUDA(cudaMemset(d_failure, 0, sizeof(bool)));

    constexpr int block_size = 256;
    const int grid_size = std::min(32768, ceil_div(N / 8, block_size));
    is_close_kernel<<<grid_size, block_size>>>(A, B, N, atol, rtol, d_failure);
    CHECK_CUDA(cudaGetLastError());

    bool h_failure;
    CHECK_CUDA(cudaMemcpy(&h_failure, d_failure, sizeof(bool), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_failure));

    return !h_failure;
}

struct KernelRecord {
    std::string name;
    std::function<void(const half *, const half *, half *, int, int, int)> fn;
};

std::vector<PerfRecord> perf(int M, int N, int K, const std::string &kernel_name) {
    half *dA, *dB;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(half)));

#ifndef HGEMM_DEBUG
    const float rsqrt_K = 1.f / std::sqrt(K);
    uniform_cuda(dA, M * K, -rsqrt_K, rsqrt_K);
    uniform_cuda(dB, K * N, -rsqrt_K, rsqrt_K);
#else
    half *A, *B;
    CHECK_CUDA(cudaMallocHost(&A, sizeof(half) * M * K));
    CHECK_CUDA(cudaMallocHost(&B, sizeof(half) * K * N));

    for (int i = 0; i < M * K; i++) {
        A[i] = __float2half(i / 100.f);
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = __float2half(i / 100.f);
    }

    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(half), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
#endif

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

#define MAKE_KERNEL(...) {#__VA_ARGS__, __VA_ARGS__}

    std::vector<KernelRecord> all_kernels{
        MAKE_KERNEL(hgemm_v1),

        MAKE_KERNEL(hgemm_v2<16, 16, 16>),

        MAKE_KERNEL(hgemm_v3<128, 128, 32, 2, 2>),

        MAKE_KERNEL(hgemm_v4<128, 128, 32, 2, 2, 2>),

        MAKE_KERNEL(hgemm_mma_v1<>),

        MAKE_KERNEL(hgemm_mma_v2<128, 128, 32, 2, 2, 3, true, 2>),

        MAKE_KERNEL(hgemm_mma_v3<128, 128, 32, 2, 2, 4, 8>),
        MAKE_KERNEL(hgemm_mma_v3<256, 128, 32, 4, 2, 4, 4>),
        MAKE_KERNEL(hgemm_mma_v3<256, 128, 32, 4, 2, 4, 8>),
        MAKE_KERNEL(hgemm_mma_v3<128, 256, 32, 2, 4, 4, 8>),

        MAKE_KERNEL(hgemm),
    };

#undef MAKE_KERNEL

    // select kernels based on args
    std::vector<KernelRecord> kernels{
        {"cublas", [handle](const half *A, const half *B, half *C, int M, int N,
                            int K) { hgemm_cublas(handle, A, B, C, M, N, K); }},
    };
    std::copy_if(all_kernels.begin(), all_kernels.end(), std::back_inserter(kernels),
                 [&kernel_name](const KernelRecord &x) { return x.name.find(kernel_name) != std::string::npos; });

    printf("----- M=%d N=%d K=%d -----\n", M, N, K);

    std::vector<PerfRecord> records;

    half *dC_ref;
    CHECK_CUDA(cudaMalloc(&dC_ref, M * N * sizeof(half)));

    hgemm_cublas(handle, dA, dB, dC_ref, M, N, K);

    for (const auto &kernel : kernels) {
        half *dC_opt;
        CHECK_CUDA(cudaMalloc(&dC_opt, M * N * sizeof(half)));
        CHECK_CUDA(cudaMemset(dC_opt, 0, M * N * sizeof(half)));

        kernel.fn(dA, dB, dC_opt, M, N, K);

        if (!is_close_cuda(dC_ref, dC_opt, M * N, 1e-3f, 1e-2f)) {
            // TODO: use gpu
            check_is_close_d(dC_ref, dC_opt, M * N, 1e-3f);
        }

        auto perf_fn = [=] { kernel.fn(dA, dB, dC_opt, M, N, K); };
        const float elapsed = timeit(perf_fn, 2, 20);

        const double tflops = (2ull * M * N * K) * 1e-12 / elapsed;
        const float bandwidth = (M * K + K * N + M * N) * sizeof(half) * 1e-9f / elapsed;

        printf("[%s] elapsed %.3f us, %.1f TFLOPS, %.3f GB/s\n", kernel.name.c_str(), elapsed * 1e6, tflops, bandwidth);

        records.emplace_back(PerfRecord(M, N, K, kernel.name, elapsed, tflops));

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

// Helper function to parse size specification
std::vector<int> parse_sizes(const std::string &spec) {
    std::vector<int> result;

    // Check if it's a range format (contains two colons)
    size_t colon1 = spec.find(':');
    size_t colon2 = spec.find(':', colon1 + 1);

    if (colon1 != std::string::npos && colon2 != std::string::npos) {
        // Range format: start:stop:step
        int start = std::stoi(spec.substr(0, colon1));
        int stop = std::stoi(spec.substr(colon1 + 1, colon2 - colon1 - 1));
        int step = std::stoi(spec.substr(colon2 + 1));

        for (int i = start; i <= stop; i += step) {
            result.emplace_back(i);
        }
    } else {
        // Comma-separated format
        std::stringstream ss(spec);
        std::string item;

        while (std::getline(ss, item, ',')) {
            result.emplace_back(std::stoi(item));
        }
    }

    return result;
}

struct Args {
    int M = 8192;
    int N = 8192;
    int K = 8192;
    std::vector<int> sizes;
    std::string kernel_name;
};

Args parse_args(int argc, char **argv) {
    Args args;

    int opt;
    while ((opt = getopt(argc, argv, "M:N:K:S:k:h")) != -1) {
        switch (opt) {
        case 'M':
            args.M = atoi(optarg);
            break;
        case 'N':
            args.N = atoi(optarg);
            break;
        case 'K':
            args.K = atoi(optarg);
            break;
        case 'S':
            args.sizes = parse_sizes(optarg);
            break;
        case 'k':
            args.kernel_name = optarg;
            break;
        case 'h':
            printf("Usage: %s [-M size] [-N size] [-K size] [-S sizes]\n", argv[0]);
            printf("Options:\n");
            printf("  -M size    Set M dimension (default: %d)\n", 8192);
            printf("  -N size    Set N dimension (default: %d)\n", 8192);
            printf("  -K size    Set K dimension (default: %d)\n", 8192);
            printf("  -S sizes   Set square matrix sizes. Format:\n");
            printf("             start:stop:step (e.g., 1024:8192:1024)\n");
            printf("             or comma-separated (e.g., 1024,2048,4096)\n");
            printf("  -k name    Filter kernel name (default: all)\n");
            printf("  -h         Show this help message\n");
            exit(EXIT_SUCCESS);
        default:
            fprintf(stderr, "Usage: %s [-M size] [-N size] [-K size] [-S sizes] [-k name]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);

#ifdef HGEMM_DEBUG
    perf(16, 8, 16);
    return 0;
#endif

    if (args.sizes.empty()) {
        perf(args.M, args.N, args.K, args.kernel_name);
        return 0;
    }

    std::vector<PerfRecord> all_records;
    for (int size : args.sizes) {
        auto records = perf(size, size, size, args.kernel_name);
        all_records.insert(all_records.end(), records.begin(), records.end());
    }
    save_result("output/hgemm_bench_square.csv", all_records);

    // fixed K to avoid split-K kernels
    // {
    //     std::vector<PerfRecord> all_records;
    //     constexpr int K = 1024;
    //     const int dims[]{1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    //     for (int d : dims) {
    //         auto records = perf(d, d, K);
    //         all_records.insert(all_records.end(), records.begin(), records.end());
    //     }

    //     save_result("output/hgemm_bench_fixk.csv", all_records);
    // }

    return 0;
}
