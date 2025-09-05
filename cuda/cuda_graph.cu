#include "common.h"
#include <nvtx3/nvToolsExt.h>

// #define NVTX_DISABLE

__global__ void short_kernel(const float *d_in, float *d_out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = 1.23 * d_in[idx];
    }
}

cudaError_t short_kernel_launch(const float *d_in, float *d_out, int N, cudaStream_t stream) {
    const int block_size = 128;
    const int grid_size = N / block_size;
    short_kernel<<<grid_size, block_size, 0, stream>>>(d_in, d_out, N);
    return cudaGetLastError();
}

void work(const float *d_in, float *d_out, int N, int NUM_KERNELS, cudaStream_t stream) {
    for (int i = 0; i < NUM_KERNELS; i++) {
        CHECK_CUDA(short_kernel_launch(d_in, d_out, N, stream));
    }
}

struct nvtx_scoped_range {
    nvtx_scoped_range(const char *name) { nvtxRangePushA(name); }
    ~nvtx_scoped_range() { nvtxRangePop(); }
};

int main() {
    const int N = 128;
    float *d_in, *d_ref, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ref, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemset(d_in, 0, N * sizeof(float)));

    const int NUM_KERNELS = 2048;

    const float naive_cost = timeit(
        [=] {
            nvtx_scoped_range marker("naive");
            work(d_in, d_ref, N, NUM_KERNELS, cudaStreamDefault);
            CHECK_CUDA(cudaStreamSynchronize(cudaStreamDefault));
        },
        2, 10);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    work(d_in, d_out, N, NUM_KERNELS, stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    const float graph_cost = timeit(
        [&] {
            nvtx_scoped_range marker("optimized");
            CHECK_CUDA(cudaGraphLaunch(instance, cudaStreamDefault));
            CHECK_CUDA(cudaStreamSynchronize(cudaStreamDefault));
        },
        2, 10);

    check_is_close_d(d_ref, d_out, N);

    printf("naive_cost: %.3f ms, graph_cost: %.3f ms, speedup: %.2f x\n", naive_cost * 1e3, graph_cost * 1e3,
           naive_cost / graph_cost);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_ref));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
