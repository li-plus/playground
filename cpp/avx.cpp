// reference: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html

#include <cassert>
#include <chrono>
#include <cinttypes>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <x86intrin.h>

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

// https://stackoverflow.com/questions/13772567/how-to-get-the-cpu-cycle-count-in-x86-64-from-c
template <typename Fn>
uint64_t count_cpu_cycles(Fn fn) {
    const auto cycle_start = __rdtsc();
    fn();
    const auto cycle_end = __rdtsc();
    return cycle_end - cycle_start;
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

    const auto cycles_avx256 = count_cpu_cycles([&] { avx256_add(a.data(), b.data(), output_avx256.data(), n); });
    const auto cycles_scalar = count_cpu_cycles([&] { scalar_add(a.data(), b.data(), output_avx256.data(), n); });

    const float elapsed_avx256 = timeit([&] { avx256_add(a.data(), b.data(), output_avx256.data(), n); }, 100);
    const float elapsed_scalar = timeit([&] { scalar_add(a.data(), b.data(), output_scalar.data(), n); }, 100);

    printf("[scalar] cpu cycles: %lu, elapsed %.3f ns\n", cycles_scalar, elapsed_scalar * 1e9f);
    printf("[avx256] cpu cycles: %lu, elapsed %.3f ns\n", cycles_avx256, elapsed_avx256 * 1e9f);

    return 0;
}