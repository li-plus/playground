#include "common.h"
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

constexpr inline bool is_power_of_2(unsigned x) { return 0 == (x & (x - 1)); }

__device__ __forceinline__ void bitonic_swap(int *pa, int *pb, bool dir) {
    const int a = *pa;
    const int b = *pb;
    if ((a < b) == dir) {
        *pa = b;
        *pb = a;
    }
}

__device__ __forceinline__ void bitonic_merge(int *s_vec, int size) {
    const bool dir = threadIdx.x & (size / 2);
#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {
        // const int pos = (threadIdx.x / stride) * (stride * 2) + threadIdx.x % stride;
        const int pos = 2 * threadIdx.x - threadIdx.x % stride;
        bitonic_swap(&s_vec[pos], &s_vec[pos + stride], dir);
        __syncthreads();
    }
}

template <int N>
__global__ void bitonic_sort_kernel(int *__restrict__ vec) {
    __shared__ int s_vec[N];

    ((int2 *)s_vec)[threadIdx.x] = ((int2 *)vec)[threadIdx.x];
    __syncthreads();

#pragma unroll
    for (int size = 2; size <= N; size *= 2) {
        bitonic_merge(s_vec, size);
    }

    ((int2 *)vec)[threadIdx.x] = ((int2 *)s_vec)[threadIdx.x];
}

template <int N>
cudaError_t bitonic_sort_cuda(int *vec) {
    static_assert(0 < N && N <= 2048 && is_power_of_2(N), "invalid array size");
    constexpr int block_size = N / 2;
    bitonic_sort_kernel<N><<<1, block_size>>>(vec);
    return cudaGetLastError();
}

int main() {
    // https://nvidia.github.io/cccl/thrust/#examples
    constexpr size_t N = 2048;
    thrust::host_vector<int> h_input(N);
    std::iota(h_input.rbegin(), h_input.rend(), 0);

    // std::sort
    thrust::host_vector<int> h_output_expect = h_input;
    std::sort(h_output_expect.begin(), h_output_expect.end());

    // thrust::sort
    thrust::device_vector<int> d_vec = h_input;
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::host_vector<int> h_output_actual = d_vec;
    CHECK(h_output_expect == h_output_actual);

    // cuda sort
    d_vec = h_input;
    CHECK_CUDA(bitonic_sort_cuda<N>(d_vec.data().get()));
    h_output_actual = d_vec;
    CHECK(h_output_expect == h_output_actual);

    // benchmark
    {
        const float elapsed = timeit([&] { std::sort(h_output_expect.begin(), h_output_expect.end()); }, 2, 10);
        printf("[cpu] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed = timeit([&] { thrust::sort(d_vec.begin(), d_vec.end()); }, 2, 10);
        printf("[thrust] elapsed %.3f us\n", elapsed * 1e6);
    }
    {
        const float elapsed = timeit([&] { CHECK_CUDA(bitonic_sort_cuda<N>(d_vec.data().get())); }, 2, 10);
        printf("[cuda] elapsed %.3f us\n", elapsed * 1e6);
    }

    return 0;
}
