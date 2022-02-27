#pragma once

#include <cuda_runtime.h>

cudaError_t sgemm_cuda(int M, int N, int K, const float *A, const float *B, float *C);
