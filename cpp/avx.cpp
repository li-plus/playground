// reference: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

#include <cassert>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <vector>

void avx256_add(const float *a, const float *b, float *output, int n) {
    int i;
    const int limit = n & ~7;
    for (i = 0; i < limit; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        __m256 sum = _mm256_add_ps(av, bv);
        _mm256_storeu_ps(&output[i], sum);
    }

    for (; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}

void scalar_add(const float *a, const float *b, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}

template <typename Fn>
float timeit(Fn fn, int n) {
    const int warmup = std::max(2, n / 100);
    for (int i = 0; i < warmup; i++) {
        fn();
    }

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        fn();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const float elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9f;
    return elapsed / n;
}

int main() {
    const int n = 1025;
    std::vector<float> a(n), b(n), output_avx256(n), output_scalar(n);
    for (int i = 0; i < n; i++) {
        a[i] = rand() / (float)RAND_MAX;
        b[i] = rand() / (float)RAND_MAX;
    }

    avx256_add(a.data(), b.data(), output_avx256.data(), n);
    scalar_add(a.data(), b.data(), output_scalar.data(), n);

    for (int i = 0; i < n; i++) {
        assert(std::abs(output_avx256[i] - output_scalar[i]) < 1e-3f);
    }

    const float elapsed_avx256 = timeit([&] { avx256_add(a.data(), b.data(), output_avx256.data(), n); }, 100);
    const float elapsed_scalar = timeit([&] { scalar_add(a.data(), b.data(), output_scalar.data(), n); }, 100);

    printf("[scalar] elapsed %.3f ns\n", elapsed_scalar * 1e9);
    printf("[avx256] elapsed %.3f ns\n", elapsed_avx256 * 1e9);

    return 0;
}