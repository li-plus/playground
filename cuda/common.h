#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdio.h>

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

#define CHECK_CUDA(status) check_cuda_status((status), __FILE__, __LINE__)

static inline void check_cuda_status(cudaError_t status, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s:%d: cuda error: %s\n", file, line, cudaGetErrorString(status));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS(status) check_cublas_status((status), __FILE__, __LINE__)

static inline void check_cublas_status(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s:%d: cublas error code: %d\n", file, line, status);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

static inline float uniform() { return rand() / (float) RAND_MAX; }

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

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static constexpr size_t KB = 1024;
static constexpr size_t MB = 1024ull * KB;
static constexpr size_t GB = 1024ull * MB;

static constexpr int WARP_SIZE = 32;

struct V100SXM2Spec {
    static constexpr float PEAK_MEM_BW = 900; // GB/s
    static constexpr float PEAK_FP32_TFLOPS = 15.7;
    static constexpr float PEAK_FP16_TFLOPS = 125;
};
