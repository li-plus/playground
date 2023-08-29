#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <stdio.h>
#include <cublas_v2.h>

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

static inline float uniform() { return (float)rand() / RAND_MAX; }

static inline float uniform(float lo, float hi) { return uniform() * (hi - lo) + lo; }

static inline float timeit(std::function<void()> fn, int warmup, int active) {
    float elapsed_ms;
    for (int i = 0; i < warmup; i++) {
        fn();
    }

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

    return elapsed_ms / active;
}

static inline bool is_close(float a, float b, float atol = 1e-5f, float rtol = 1e-8f) {
    return std::abs(a - b) < atol + rtol * std::abs(b);
}

static constexpr size_t KB = 1024;
static constexpr size_t MB = 1024ull * KB;
static constexpr size_t GB = 1024ull * MB;
