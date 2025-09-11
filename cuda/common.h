#pragma once

#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <sstream>
#include <stdio.h>
#include <vector>

namespace cg = cooperative_groups;

class LogMessageFatal {
  public:
    LogMessageFatal(const char *file, int line) { oss_ << file << ':' << line << ' '; }
    [[noreturn]] ~LogMessageFatal() noexcept(false) { throw std::runtime_error(oss_.str()); }
    std::ostringstream &stream() { return oss_; }

  private:
    std::ostringstream oss_;
};

#define THROW LogMessageFatal(__FILE__, __LINE__).stream()
#define CHECK(cond)                                                                                                    \
    if (!(cond))                                                                                                       \
    THROW << "check failed (" #cond ") "

#define CHECK_CUDA(expr)                                                                                               \
    do {                                                                                                               \
        const cudaError_t status = (expr);                                                                             \
        CHECK(status == cudaSuccess) << "cuda error: " << cudaGetErrorString(status);                                  \
    } while (false)

#define CHECK_CUBLAS(expr)                                                                                             \
    do {                                                                                                               \
        const cublasStatus_t status = (expr);                                                                          \
        CHECK(status == CUBLAS_STATUS_SUCCESS)                                                                         \
            << "cublas error: [" << cublasGetStatusName(status) << "] " << cublasGetStatusString(status);              \
    } while (false)

#define CHECK_CUDNN(expr)                                                                                              \
    do {                                                                                                               \
        const cudnnStatus_t status = (expr);                                                                           \
        CHECK(status == CUDNN_STATUS_SUCCESS) << "cudnn error: " << cudnnGetErrorString(status);                       \
    } while (false)

static inline float uniform() { return rand() / (float)RAND_MAX; }

static inline float uniform(float lo, float hi) { return uniform() * (hi - lo) + lo; }

template <typename Fn>
static inline float timeit(Fn fn, int warmup, int active) {
    for (int i = 0; i < warmup; i++) {
        fn();
    }

    float elapsed_ms;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < active; i++) {
        fn();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return elapsed_ms * 1e-3f / active;
}

static inline bool is_close(float a, float b, float atol = 1e-5f, float rtol = 1e-8f) {
    return std::abs(a - b) < atol + rtol * std::abs(b);
}

static inline void check_is_close(const float *a, const float *b, size_t n, float atol = 1e-5f, float rtol = 1e-8f) {
    float max_abs_diff = -INFINITY;
    int diff_cnt = 0;
    for (size_t i = 0; i < n; i++) {
        if (!is_close(a[i], b[i], atol, rtol)) {
            diff_cnt++;
            max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
        }
    }
    CHECK(diff_cnt == 0) << "max abs diff: " << max_abs_diff << ", total diff cnt: " << diff_cnt;
}

static inline void check_is_close(const half *a, const half *b, size_t n, float atol = 1e-5f, float rtol = 1e-8f) {
    std::vector<float> f_a(n), f_b(n);
    std::transform(a, a + n, f_a.begin(), [](half v) { return __half2float(v); });
    std::transform(b, b + n, f_b.begin(), [](half v) { return __half2float(v); });
    check_is_close(f_a.data(), f_b.data(), n, atol, rtol);
}

template <typename T>
static inline void check_is_close_d(const T *d_a, const T *d_b, size_t n, float atol = 1e-5f, float rtol = 1e-8f) {
    T *h_a, *h_b;
    CHECK_CUDA(cudaMallocHost(&h_a, n * sizeof(T)));
    CHECK_CUDA(cudaMallocHost(&h_b, n * sizeof(T)));

    CHECK_CUDA(cudaMemcpy(h_a, d_a, n * sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_b, d_b, n * sizeof(T), cudaMemcpyDeviceToHost));

    check_is_close(h_a, h_b, n, atol, rtol);

    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
}

static inline constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

static constexpr unsigned FULL_MASK = 0xffffffff;
static constexpr int WARP_SIZE = 32;

template <int warp_size = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(FULL_MASK, v, mask);
    }
    return v;
}

template <int block_size, bool all>
__device__ __forceinline__ float block_reduce_sum(float v) {
    v = warp_reduce_sum(v);
    constexpr int num_warps = block_size / WARP_SIZE;
    if constexpr (num_warps > 1) {
        __shared__ float shm[num_warps];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_sum<num_warps>((lane_id < num_warps) ? shm[lane_id] : 0.f);
        if constexpr (all && num_warps < WARP_SIZE) {
            v = __shfl_sync(FULL_MASK, v, 0);
        }
    }
    return v;
}

__device__ __forceinline__ float cg_warp_reduce_sum(float v) {
    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    return cg::reduce(warp, v, cg::plus<float>());
}

template <int block_size, bool all>
__device__ __forceinline__ float cg_block_reduce_sum(float v) {
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    v = cg::reduce(warp, v, cg::plus<float>());
    constexpr int num_warps = block_size / WARP_SIZE;
    if constexpr (num_warps > 1) {
        __shared__ float shm[num_warps];
        const int warp_id = warp.meta_group_rank();
        const int lane_id = warp.thread_rank();
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        block.sync();
        auto tile = cg::tiled_partition<num_warps>(block);
        v = cg::reduce(tile, (lane_id < num_warps) ? shm[lane_id] : 0.f, cg::plus<float>());
        if constexpr (all && num_warps < WARP_SIZE) {
            v = warp.shfl(v, 0);
        }
    }
    return v;
}

template <int block_size>
__device__ __forceinline__ float shm_block_reduce_sum(float v) {
    __shared__ float shm[block_size];

    shm[threadIdx.x] = v;
    __syncthreads();

#pragma unroll
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shm[0];
}

template <int warp_size = 32>
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = warp_size / 2; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, mask, 32));
    }
    return v;
}

template <int block_size, bool all>
__device__ __forceinline__ float block_reduce_max(float v) {
    v = warp_reduce_max(v);
    constexpr int num_warps = block_size / WARP_SIZE;
    if constexpr (num_warps > 1) {
        __shared__ float shm[num_warps];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shm[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_max<num_warps>((lane_id < num_warps) ? shm[lane_id] : -INFINITY);
        if constexpr (all && num_warps < WARP_SIZE) {
            v = __shfl_sync(FULL_MASK, v, 0);
        }
    }
    return v;
}
