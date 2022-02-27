#include "gemm.h"

#include <cmath>
#include <cublas_v2.h>
#include <iostream>

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

template <typename Func>
float cuda_timeit(Func f, int repeat = 1, int warmup = 0) {
    for (int i = 0; i < warmup; i++) {
        f();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_cost;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        f();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&ms_cost, start, stop));
    return ms_cost / repeat;
}

class CudaGemm {
  public:
    cudaError_t sgemm(int M, int N, int K, const float *dA, const float *dB, float *dC) {
        return sgemm_cuda(M, N, K, dA, dB, dC);
    }
};

class CublasGemm {
  public:
    CublasGemm() { CHECK_CUBLAS(cublasCreate(&_cublas_handle)); }
    CublasGemm(const CublasGemm &other) = delete;
    CublasGemm &operator=(const CublasGemm &other) = delete;
    ~CublasGemm() { CHECK_CUBLAS(cublasDestroy(_cublas_handle)); }

    cublasStatus_t sgemm(int M, int N, int K, const float *dA, const float *dB, float *dC) {
        const float alpha = 1;
        const float beta = 0;
        return cublasSgemm(_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    }

  private:
    cublasHandle_t _cublas_handle;
};

void perf(int M, int N, int K) {
    // make data
    float *A = (float *)malloc(sizeof(float) * M * K);
    float *B = (float *)malloc(sizeof(float) * K * N);
    for (int i = 0; i < M * K; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = rand() / (float)RAND_MAX;
    }
    float *dA;
    float *dB;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuda impl
    CudaGemm cuda_gemm;
    float *C1 = (float *)malloc(M * N * sizeof(float));
    float *dC1;
    CHECK_CUDA(cudaMalloc(&dC1, M * N * sizeof(float)));
    CHECK_CUDA(cuda_gemm.sgemm(M, N, K, dA, dB, dC1));
    CHECK_CUDA(cudaMemcpy(C1, dC1, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cublas impl
    CublasGemm cublas_gemm;
    float *C2 = (float *)malloc(M * N * sizeof(float));
    float *dC2;
    CHECK_CUDA(cudaMalloc(&dC2, M * N * sizeof(float)));
    CHECK_CUBLAS(cublas_gemm.sgemm(M, N, K, dA, dB, dC2));
    CHECK_CUDA(cudaMemcpy(C2, dC2, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // check correctness
    bool is_correct = true;
    for (int i = 0; i < M * N; i++) {
        float abs_err = std::abs(C1[i] - C2[i]);
        float rel_err = abs_err / std::max(std::abs(C1[i]), std::abs(C2[i]));
        constexpr float rel_tol = 1e-5;
        if (rel_err >= rel_tol) {
            int x = i % N;
            int y = i / N;
            printf("Result error at (%d, %d): c1=%f vs c2=%f (abs_err=%f, rel_err=%f)\n", y, x, C1[i], C2[i], abs_err,
                   rel_err);
            is_correct = false;
            break;
        }
    }

    if (is_correct) {
        // perf cuda
        auto cuda_f = [&] { cuda_gemm.sgemm(M, N, K, dA, dB, dC1); };
        float cuda_ms = cuda_timeit(cuda_f, 100, 10);
        // perf cublas
        auto cublas_f = [&] { cublas_gemm.sgemm(M, N, K, dA, dB, dC2); };
        float cublas_ms = cuda_timeit(cublas_f, 100, 10);
        // print result
        printf("[M=%4d, N=%4d, K=%4d] cuda: %f ms, cublas: %f ms, speedup: %.1f%%\n", M, N, K, cuda_ms, cublas_ms,
               100 * cublas_ms / cuda_ms);
    }

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