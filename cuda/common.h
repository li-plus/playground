#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdio.h>

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
        cudaError_t status = (expr);                                                                                   \
        CHECK(status == cudaSuccess) << "cuda error: " << cudaGetErrorString(status);                                  \
    } while (false)

#define CHECK_CUBLAS(expr)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = (expr);                                                                                \
        CHECK(status == CUBLAS_STATUS_SUCCESS)                                                                         \
            << "cublas error: [" << cublasGetStatusName(status) << "] " << cublasGetStatusString(status);              \
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
    for (size_t i = 0; i < n; i++) {
        CHECK(is_close(a[i], b[i], atol, rtol)) << a[i] << " vs " << b[i];
    }
}

static inline void check_is_close(const half *a, const half *b, size_t n, float atol = 1e-5f, float rtol = 1e-8f) {
    for (size_t i = 0; i < n; i++) {
        CHECK(is_close((float)a[i], (float)b[i], atol, rtol)) << (float)a[i] << " vs " << (float)b[i];
    }
}

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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
