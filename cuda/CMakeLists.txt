cmake_minimum_required(VERSION 3.18)
project(CUDA_LAB LANGUAGES CXX CUDA)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(PTX_VERBOSE "" OFF)
if (PTX_VERBOSE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xptxas=-v")
endif ()

set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DNDEBUG -O3 --use_fast_math")

add_executable(add add.cu)

add_executable(gemv gemv.cu)
target_link_libraries(gemv ${CUDA_CUBLAS_LIBRARIES})

add_executable(gemm gemm.cu)
target_link_libraries(gemm ${CUDA_CUBLAS_LIBRARIES})

# clang-format
file(GLOB SOURCES *.cpp *.h *.cu *.cuh)
add_custom_target(lint COMMAND clang-format -i ${SOURCES})
