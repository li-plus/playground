// $ g++ gemm.cpp -o gemm -O3 -std=c++11 -march=native && ./gemm

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

void gemm(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    const int M = 512;
    const int N = 512;
    const int K = 512;
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    for (auto &x : A) {
        x = rand() / float(RAND_MAX);
    }
    for (auto &x : B) {
        x = rand() / float(RAND_MAX);
    }

    const int warmup = 2;
    const int active = 10;
    for (int i = 0; i < warmup; i++) {
        gemm(A.data(), B.data(), C.data(), M, N, K);
    }

    const auto start = std::chrono::system_clock::now();
    for (int i = 0; i < active; i++) {
        gemm(A.data(), B.data(), C.data(), M, N, K);
    }
    const auto end = std::chrono::system_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (float)active;
    const float gflops = 2.0f * M * N * K / (elapsed * 1e-6f) * 1e-9f;
    std::cout << "Elapsed time: " << elapsed * 1e-3 << " ms, " << gflops << " GFLOPS" << "\n";

    return 0;
}