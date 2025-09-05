/*
See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleIPC/simpleIPC.cu
Usage: mpirun -np 8 bin/ipc
*/

#include "common.h"
#include <mpi.h>
#include <vector>

#define CHECK_MPI(call)                                                                                                \
    do {                                                                                                               \
        const int status = (call);                                                                                     \
        if (status != MPI_SUCCESS) {                                                                                   \
            char error_string[MPI_MAX_ERROR_STRING];                                                                   \
            int error_length;                                                                                          \
            MPI_Error_string(status, error_string, &error_length);                                                     \
            THROW << "MPI error: " << error_string;                                                                    \
        }                                                                                                              \
    } while (false)

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
                                      uint32_t **__restrict__ peers_flag, uint32_t flag_value, int rank, int world_size,
                                      int N) {
    // flag: [world_size, world_size]

    uint32_t *local_flag = peers_flag[rank];

    const int peer_rank = (blockIdx.y + rank) % world_size;
    int *peer_output = peers_output[peer_rank] + rank * N;
    for (int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += 4 * gridDim.x * blockDim.x) {
        *(float4 *)&peer_output[i] = *(float4 *)&input[i];
    }

    __shared__ int prev_flag;
    if (threadIdx.x == 0) {
        prev_flag = atomicAdd(local_flag + rank, 1);
    }
    __syncthreads();

    if (prev_flag == flag_value - 1) {
        if (threadIdx.x < world_size) {
            st_flag_release(flag_value, peers_flag[threadIdx.x] + rank);

            while (ld_flag_acquire(local_flag + threadIdx.x) != flag_value) {
            }
        }
    }
}

void ipc_all_gather_cuda(const int *input, int **peers_output, uint32_t **peers_flag, uint32_t *flag, int run_count,
                         int rank, int world_size, int N) {
    constexpr int block_size = 128;
    const dim3 grid_size((N / 4 + block_size - 1) / block_size, world_size);
    uint32_t flag_value = (run_count + 1) * grid_size.x * grid_size.y;
    ipc_all_gather_kernel<<<grid_size, block_size>>>(input, peers_output, peers_flag, flag_value, rank, world_size, N);
}

int main(int argc, char **argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));

    int world_size, rank;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    printf("[rank %d] initialized world size %d\n", rank, world_size);

    CHECK_CUDA(cudaSetDevice(rank));

    const int N = 2 * 1024 * 1024;

    int *h_output;
    CHECK_CUDA(cudaMallocHost(&h_output, world_size * N * sizeof(int)));

    int *d_input;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_input, rank, N * sizeof(int)));

    std::vector<int *> d_output_h_vec(world_size);
    CHECK_CUDA(cudaMalloc(&d_output_h_vec[rank], world_size * N * sizeof(int)));

    std::vector<uint32_t *> d_flag_h_vec(world_size);
    CHECK_CUDA(cudaMalloc(&d_flag_h_vec[rank], world_size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_flag_h_vec[rank], 0, world_size * sizeof(uint32_t)));

    // ipc mem
    std::vector<cudaIpcMemHandle_t> mem_handles(world_size);
    CHECK_CUDA(cudaIpcGetMemHandle(mem_handles.data() + rank, d_output_h_vec[rank]));
    CHECK_MPI(MPI_Allgather(mem_handles.data() + rank, sizeof(cudaIpcMemHandle_t), MPI_BYTE, mem_handles.data(),
                            sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(
                cudaIpcOpenMemHandle((void **)&d_output_h_vec[i], mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }
    int **d_output_d_vec;
    CHECK_CUDA(cudaMalloc(&d_output_d_vec, world_size * sizeof(void *)));
    CHECK_CUDA(
        cudaMemcpyAsync(d_output_d_vec, d_output_h_vec.data(), world_size * sizeof(void *), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaIpcGetMemHandle(mem_handles.data() + rank, d_flag_h_vec[rank]));
    CHECK_MPI(MPI_Allgather(mem_handles.data() + rank, sizeof(cudaIpcMemHandle_t), MPI_BYTE, mem_handles.data(),
                            sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcOpenMemHandle((void **)&d_flag_h_vec[i], mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
        }
    }
    uint32_t **d_flag_d_vec;
    CHECK_CUDA(cudaMalloc(&d_flag_d_vec, world_size * sizeof(void *)));
    CHECK_CUDA(cudaMemcpyAsync(d_flag_d_vec, d_flag_h_vec.data(), world_size * sizeof(void *), cudaMemcpyHostToDevice));

    // ipc events
    // std::vector<cudaEvent_t> events(world_size);
    // std::vector<cudaIpcEventHandle_t> event_handles(world_size);
    // CHECK_CUDA(cudaEventCreate(&events[rank], cudaEventDisableTiming | cudaEventInterprocess));
    // CHECK_CUDA(cudaIpcGetEventHandle(&event_handles[rank], events[rank]));
    // MPI_Allgather(event_handles.data() + rank, sizeof(cudaIpcEventHandle_t), MPI_BYTE, event_handles.data(),
    //               sizeof(cudaIpcEventHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    // for (int i = 0; i < world_size; i++) {
    //     if (i != rank) {
    //         CHECK_CUDA(cudaIpcOpenEventHandle(&events[i], event_handles[i]));
    //     }
    // }

    int run_count = 0;

    // run & check
    ipc_all_gather_cuda(d_input, d_output_d_vec, d_flag_d_vec, d_flag_h_vec[rank], run_count++, rank, world_size, N);
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
            ipc_all_gather_cuda(d_input, d_output_d_vec, d_flag_d_vec, d_flag_h_vec[rank], run_count++, rank,
                                world_size, N);
        },
        10, 1000);
    const float bus_bandwidth = (world_size - 1) * N * sizeof(int) / 1e9f / elapsed;
    printf("[rank %d] [cuda] elapsed %.3f us, (uni-directional) bus_bandwidth %.3f GB/s\n", rank, elapsed * 1e6f,
           bus_bandwidth);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

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

    CHECK_MPI(MPI_Finalize());

    return 0;
}