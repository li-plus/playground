cuda_add_library(gemm gemm.cu)
target_link_libraries(gemm cublas)

add_executable(gemm_bench gemm_bench.cpp)
target_link_libraries(gemm_bench gemm)
