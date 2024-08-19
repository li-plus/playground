#include "common.h"

__global__ void loop_cuda_kernel() {
    while (true) {
    }
}

void loop_cuda() { loop_cuda_kernel<<<1, 1>>>(); }

int main() {
    loop_cuda();
    printf("infinite loop\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}
