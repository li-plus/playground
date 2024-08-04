#include "common.h"

__global__ void noop_cuda_kernel() {}

void noop_cuda() { noop_cuda_kernel<<<1, 1>>>(); }

int main() {
    const float elapsed = timeit(noop_cuda, 100, 10000);
    printf("kernel launch cost: %.3f us\n", elapsed * 1e6f); // ~2us per kernel launch
    return 0;
}
