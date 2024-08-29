/*
See https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleIPC/simpleIPC.cu
Usage: mpirun -np 8 bin/ipc
*/

#include "common.h"
#include <mpi.h>
#include <vector>

__global__ void ipc_all_gather_kernel(const int *__restrict__ input, int **__restrict__ peers_output, int rank, int N) {
    // volatile clock_t start_clock = clock();
    // volatile clock_t clock_offset = 0;
    // while (clock_offset < rank * 10000000ull) {
    //     clock_offset = clock() - start_clock;
    // }
    // printf("[rank %d] kernel start\n", rank);

    int *peer_output = peers_output[blockIdx.y] + rank * N;
    for (int i = 4 * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += 4 * gridDim.x * blockDim.x) {
        *(float4 *)&peer_output[i] = *(float4 *)&input[i];
    }
}

void ipc_all_gather_cuda(const int *input, int **peers_output, int rank, int world_size, int N) {
    constexpr int block_size = 1024;
    const dim3 grid_size((N / 4 + block_size - 1) / block_size, world_size);
    ipc_all_gather_kernel<<<grid_size, block_size>>>(input, peers_output, rank, N);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("[rank %d] initialized world size %d\n", rank, world_size);

    CHECK_CUDA(cudaSetDevice(rank));

    const int N = 512 * 1024;

    int *h_output;
    CHECK_CUDA(cudaMallocHost(&h_output, world_size * N * sizeof(int)));

    int *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, world_size * N * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_input, rank, N * sizeof(int)));

    // ipc mem
    std::vector<cudaIpcMemHandle_t> mem_handles(world_size);
    CHECK_CUDA(cudaIpcGetMemHandle(mem_handles.data() + rank, d_output));
    MPI_Allgather(mem_handles.data() + rank, sizeof(cudaIpcMemHandle_t), MPI_BYTE, mem_handles.data(),
                  sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    std::vector<void *> d_output_h_vec(world_size);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcOpenMemHandle(&d_output_h_vec[i], mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
        } else {
            d_output_h_vec[i] = d_output;
        }
    }
    void **d_output_d_vec;
    CHECK_CUDA(cudaMalloc(&d_output_d_vec, world_size * sizeof(void *)));
    CHECK_CUDA(
        cudaMemcpyAsync(d_output_d_vec, d_output_h_vec.data(), world_size * sizeof(void *), cudaMemcpyHostToDevice));

    // ipc events
    std::vector<cudaEvent_t> events(world_size);
    std::vector<cudaIpcEventHandle_t> event_handles(world_size);
    CHECK_CUDA(cudaEventCreate(&events[rank], cudaEventDisableTiming | cudaEventInterprocess));
    CHECK_CUDA(cudaIpcGetEventHandle(event_handles.data() + rank, events[rank]));
    MPI_Allgather(event_handles.data() + rank, sizeof(cudaIpcEventHandle_t), MPI_BYTE, event_handles.data(),
                  sizeof(cudaIpcEventHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcOpenEventHandle(&events[i], event_handles[i]));
        }
    }

    ipc_all_gather_cuda(d_input, (int **)d_output_d_vec, rank, world_size, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(events[rank]));
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < world_size; i++) {
        CHECK_CUDA(cudaStreamWaitEvent(0, events[i]));
    }
    CHECK_CUDA(cudaMemcpy(h_output, d_output, world_size * N * sizeof(int), cudaMemcpyDeviceToHost));

    int *h_output_ref;
    CHECK_CUDA(cudaMallocHost(&h_output_ref, world_size * N * sizeof(int)));
    for (int i = 0; i < world_size; i++) {
        memset(h_output_ref + i * N, i, N * sizeof(int));
    }
    CHECK(memcmp(h_output, h_output_ref, world_size * N * sizeof(int)) == 0);

    const float elapsed = timeit(
        [&] {
            ipc_all_gather_cuda(d_input, (int **)d_output_d_vec, rank, world_size, N);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(events[rank]));
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < world_size; i++) {
                CHECK_CUDA(cudaStreamWaitEvent(0, events[i]));
            }
        },
        10, 100);
    const float bandwidth = world_size * N * sizeof(int) / 1e9f / elapsed;
    printf("[cuda] elapsed %.3f us, bandwidth %.3f GB/s\n", elapsed * 1e6f, bandwidth);

    // clean up
    for (int i = 0; i < world_size; i++) {
        if (i != rank) {
            CHECK_CUDA(cudaIpcCloseMemHandle(d_output_h_vec[i]));
        }
    }

    for (int i = 0; i < world_size; i++) {
        CHECK_CUDA(cudaEventDestroy(events[i]));
    }

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_output_d_vec));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFreeHost(h_output_ref));

    MPI_Finalize();

    return 0;
}