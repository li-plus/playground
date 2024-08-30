/*
See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleIPC/simpleIPC.cu
Usage: mpirun -np 8 bin/ipc
*/

#include "common.h"
#include <mpi.h>
#include <vector>

// From https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu
static inline __device__ void st_flag_release(uint32_t const &flag, uint32_t *flag_addr) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static inline __device__ uint32_t ld_flag_acquire(uint32_t *flag_addr) {
    uint32_t flag;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

__global__ void ipc_all_gather_kernel(const int *__restrict__ input, int **__restrict__ peers_output,
                                      uint32_t **__restrict__ flag, int rank, int world_size, int N) {
    constexpr bool simulate_unbalance = false;
    if constexpr (simulate_unbalance) {
        volatile clock_t start_clock = clock();
        volatile clock_t clock_offset = 0;
        while (clock_offset < rank * 1'000'000'000ull) {
            clock_offset = clock() - start_clock;
        }
        if (threadIdx.x + blockIdx.x + blockIdx.y == 0) {
            printf("[rank %d] kernel start\n", rank);
        }
    }

    const int peer_rank = (blockIdx.y + rank) % world_size;
    int *peer_output = peers_output[peer_rank] + rank * N;
    for (int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += 4 * gridDim.x * blockDim.x) {
        *(float4 *)&peer_output[i] = *(float4 *)&input[i];
    }

    __shared__ int prev_blocks;
    if (threadIdx.x == 0) {
        prev_blocks = atomicAdd(flag[rank], 1);
    }
    __syncthreads();

    if (prev_blocks == gridDim.y * gridDim.x - 1) {
        if (threadIdx.x < world_size) {
            uint32_t *peer_flag = flag[threadIdx.x];
            while (ld_flag_acquire(peer_flag) != gridDim.y * gridDim.x) {
            }
        }
    }
}

void ipc_all_gather_cuda(const int *input, int **peers_output, uint32_t **peers_flag, int rank, int world_size, int N) {
    constexpr int block_size = 32;
    const dim3 grid_size((N / 4 + block_size - 1) / block_size, world_size);
    ipc_all_gather_kernel<<<grid_size, block_size>>>(input, peers_output, peers_flag, rank, world_size, N);
    CHECK_CUDA(cudaGetLastError());
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[rank %d] initialized world size %d\n", rank, world_size);

    CHECK_CUDA(cudaSetDevice(rank));

    const int N = 1024 * 1024;

    int *h_output;
    CHECK_CUDA(cudaMallocHost(&h_output, world_size * N * sizeof(int)));

    int *d_input;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_input, rank, N * sizeof(int)));

    std::vector<void *> d_output_h_vec(world_size);
    CHECK_CUDA(cudaMalloc(&d_output_h_vec[rank], world_size * N * sizeof(int)));

    std::vector<void *> d_flag_h_vec(world_size);
    CHECK_CUDA(cudaMalloc(&d_flag_h_vec[rank], sizeof(uint32_t)));

    // ipc mem
    std::vector<cudaIpcMemHandle_t> mem_handles(world_size);
    CHECK_CUDA(cudaIpcGetMemHandle(mem_handles.data() + rank, d_output_h_vec[rank]));
    MPI_Allgather(mem_handles.data() + rank, sizeof(cudaIpcMemHandle_t), MPI_BYTE, mem_handles.data(),
                  sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcOpenMemHandle(&d_output_h_vec[i], mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }
    void **d_output_d_vec;
    CHECK_CUDA(cudaMalloc(&d_output_d_vec, world_size * sizeof(void *)));
    CHECK_CUDA(
        cudaMemcpyAsync(d_output_d_vec, d_output_h_vec.data(), world_size * sizeof(void *), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaIpcGetMemHandle(mem_handles.data() + rank, d_flag_h_vec[rank]));
    MPI_Allgather(mem_handles.data() + rank, sizeof(cudaIpcMemHandle_t), MPI_BYTE, mem_handles.data(),
                  sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcOpenMemHandle(&d_flag_h_vec[i], mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }
    void **d_flag_d_vec;
    CHECK_CUDA(cudaMalloc(&d_flag_d_vec, world_size * sizeof(void *)));
    CHECK_CUDA(cudaMemcpyAsync(d_flag_d_vec, d_flag_h_vec.data(), world_size * sizeof(void *), cudaMemcpyHostToDevice));

    // run & check
    CHECK_CUDA(cudaMemsetAsync(d_flag_h_vec[rank], 0, sizeof(uint32_t)));
    ipc_all_gather_cuda(d_input, (int **)d_output_d_vec, (uint32_t **)d_flag_d_vec, rank, world_size, N);
    CHECK_CUDA(cudaMemcpy(h_output, d_output_h_vec[rank], world_size * N * sizeof(int), cudaMemcpyDeviceToHost));

    int *h_output_ref;
    CHECK_CUDA(cudaMallocHost(&h_output_ref, world_size * N * sizeof(int)));
    for (int i = 0; i < world_size; i++) {
        memset(h_output_ref + i * N, i, N * sizeof(int));
    }
    CHECK(memcmp(h_output, h_output_ref, world_size * N * sizeof(int)) == 0);

    // benchmark
    const float elapsed = timeit(
        [&] {
            CHECK_CUDA(cudaMemsetAsync(d_flag_h_vec[rank], 0, sizeof(uint32_t)));
            ipc_all_gather_cuda(d_input, (int **)d_output_d_vec, (uint32_t **)d_flag_d_vec, rank, world_size, N);
        },
        10, 100);
    const float bus_bandwidth = (world_size - 1) * N * sizeof(int) / 1e9f / elapsed;
    printf("[cuda] elapsed %.3f us, (uni-directional) bus_bandwidth %.3f GB/s\n", elapsed * 1e6f, bus_bandwidth);

    // clean up
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcCloseMemHandle(d_output_h_vec[i]));
        }
    }

    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_h_vec[rank]));
    CHECK_CUDA(cudaFree(d_flag_h_vec[rank]));
    CHECK_CUDA(cudaFree(d_output_d_vec));
    CHECK_CUDA(cudaFree(d_flag_d_vec));
    CHECK_CUDA(cudaFreeHost(h_output_ref));

    MPI_Finalize();

    return 0;
}