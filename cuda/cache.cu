#include "common.h"

template <typename ld_t, typename st_t>
__global__ void cpy_kernel(void *dst, const void *src, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count / sizeof(float4); i += gridDim.x * blockDim.x) {
        st_t()(&((float4 *)dst)[i], ld_t()(&((float4 *)src)[i]));
    }
}

template <typename ld_t, typename st_t>
void cpy_cuda(void *dst, const void *src, size_t count) {
    constexpr int num_threads = 1024;
    const int num_blocks = (count + num_threads - 1) / (16 * num_threads);
    cpy_kernel<ld_t, st_t><<<num_blocks, num_threads>>>(dst, src, count);
}

struct ld_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return *ptr; // ld.global.v4.u32
    }
};

struct ldg_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldg(ptr); // ld.global.nc.v4.f32
    }
};

struct ldca_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldca(ptr); // ld.global.ca.v4.f32
    }
};

struct ldcg_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldcg(ptr); // ld.global.cg.v4.f32
    }
};

struct ldcs_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldcs(ptr); // ld.global.cs.v4.f32
    }
};

struct ldlu_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldlu(ptr); // ld.global.lu.v4.f32
    }
};

struct ldcv_t {
    template <typename T>
    __device__ T operator()(const T *ptr) {
        return __ldcv(ptr); // ld.global.cv.v4.f32
    }
};

struct st_t {
    template <typename T>
    __device__ void operator()(T *ptr, T value) {
        *ptr = value; // st.global.v4.u32
    }
};

struct stwb_t {
    template <typename T>
    __device__ void operator()(T *ptr, T value) {
        __stwb(ptr, value); // st.global.wb.v4.f32
    }
};

struct stcg_t {
    template <typename T>
    __device__ void operator()(T *ptr, T value) {
        __stcg(ptr, value); // st.global.cg.v4.f32
    }
};

struct stcs_t {
    template <typename T>
    __device__ void operator()(T *ptr, T value) {
        __stcs(ptr, value); // st.global.cs.v4.f32
    }
};

struct stwt_t {
    template <typename T>
    __device__ void operator()(T *ptr, T value) {
        __stwt(ptr, value); // st.global.wt.v4.f32
    }
};

void test(size_t N) {
    printf("===== N = %zu =====\n", N);

    char *h_a, *h_b;
    CHECK_CUDA(cudaMallocHost(&h_a, N, cudaHostAllocDefault));
    CHECK_CUDA(cudaMallocHost(&h_b, N, cudaHostAllocDefault));

    char *d_a, *d_b;
    CHECK_CUDA(cudaMalloc(&d_a, N));
    CHECK_CUDA(cudaMalloc(&d_b, N));

    memset(h_b, 0x9c, N);
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, N, cudaMemcpyHostToDevice));

    // compute
    auto run_and_check = [&](decltype(cpy_cuda<ld_t, st_t>) fn) {
        CHECK_CUDA(cudaMemsetAsync(d_b, 0x00, N));
        fn(d_b, d_a, N);
        CHECK_CUDA(cudaMemcpy(h_b, d_b, N, cudaMemcpyDeviceToHost));
        CHECK(memcmp(h_a, h_b, N) == 0);
    };
    run_and_check(cpy_cuda<ld_t, st_t>);
    run_and_check(cpy_cuda<ldg_t, st_t>);
    run_and_check(cpy_cuda<ldca_t, st_t>);
    run_and_check(cpy_cuda<ldcg_t, stcg_t>);
    run_and_check(cpy_cuda<ldcs_t, stcs_t>);
    run_and_check(cpy_cuda<ldlu_t, st_t>);
    run_and_check(cpy_cuda<ldcv_t, st_t>);

    run_and_check(cpy_cuda<ld_t, stwb_t>);
    run_and_check(cpy_cuda<ld_t, stwt_t>);

    auto benchmark = [&](decltype(cpy_cuda<ld_t, st_t>) fn, const char *name) {
        const float elapsed = timeit([=] { fn(d_b, d_a, N); }, 10, 100);
        const float bandwidth = N / 1e9f / elapsed;
        printf("[%4s] elapsed %.3f us, bandwidth %.3f GB/s\n", name, elapsed * 1e6f, bandwidth);
    };
    benchmark(cpy_cuda<ld_t, st_t>, "dflt");
    benchmark(cpy_cuda<ldg_t, st_t>, "ldg");
    benchmark(cpy_cuda<ldca_t, st_t>, "ldca");
    benchmark(cpy_cuda<ldcg_t, stcg_t>, "ldcg");
    benchmark(cpy_cuda<ldcs_t, stcs_t>, "ldcs");
    benchmark(cpy_cuda<ldlu_t, st_t>, "ldlu");
    benchmark(cpy_cuda<ldcv_t, st_t>, "ldcv");

    benchmark(cpy_cuda<ld_t, stwb_t>, "stwb");
    benchmark(cpy_cuda<ld_t, stwt_t>, "stwt");

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
}

int main() {
    test(1024 * 64);
    test(1024 * 256);
    test(1024 * 1024);
    test(1024 * 1024 * 4);
    test(1024 * 1024 * 16);
    test(1024 * 1024 * 64);
    test(1024 * 1024 * 256);
    return 0;
}
