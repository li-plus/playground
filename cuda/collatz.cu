#include "common.h"

template <int tile_size>
__global__ void collatz_kernel() {
#define DO_CHECK

    const uint64_t start_k = tile_size * ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);

#pragma unroll
    for (int i = 0; i < tile_size; i++) {
        const uint64_t k = start_k + i;
        const uint64_t num = 4 * k + 3;
        uint64_t x = 9 * k + 8;
        x >>= __clzll(__brevll(x));
        while (x > num) {
            x = x * 3 + 1;
            x >>= __clzll(__brevll(x));
        }

#ifdef DO_CHECK
        if (num == 27) {
            // https://en.wikipedia.org/wiki/Collatz_conjecture
            assert(x == 23);
        }
#endif

        if (x == num) {
            printf("impossible! n=%ld\n", num);
        }
    }
}

cudaError_t collatz_cuda(uint64_t max_n) {
    constexpr int tile_size = 128;
    const uint64_t block_size = 64;
    const uint64_t grid_size = max_n / block_size / tile_size / 4;
    printf("Launching collatz_kernel with grid=%ld, block=%ld\n", grid_size, block_size);
    collatz_kernel<tile_size><<<grid_size, block_size>>>();
    return cudaGetLastError();
}

int main() {
    const uint64_t max_n = 1ull << 40;
    const float elapsed = timeit([&] { CHECK_CUDA(collatz_cuda(max_n)); }, 0, 1);
    const float sample_per_sec = max_n / elapsed / 1e9;
    printf("elapsed %.3f s, %.3f G sample/s\n", elapsed, sample_per_sec);
    return 0;
}
