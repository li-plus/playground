cmake_minimum_required(VERSION 3.12)
project(gemm_lab)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 70 75)

option(PTX_VERBOSE "" OFF)
if (PTX_VERBOSE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ptxas-options=-v")
endif ()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE} -DNDEBUG -O3 --use_fast_math")

add_subdirectory(src)

# clang-format
file(GLOB_RECURSE SOURCES src/*.cpp src/*.h src/*.cu src/*.cuh)
add_custom_target(format COMMAND clang-format -i ${SOURCES})
