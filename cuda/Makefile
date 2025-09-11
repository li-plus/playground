TARGETS := noop fuse memcpy reduce_sum reduce_sum_dim float4 sgemm sgemv hgemm hgemv sched coalesce transpose loop cache softmax gemm_bias_gelu vote sort conv sm_sched collatz cuda_graph cp_async

CUDAFLAGS := -std=c++17 -gencode arch=compute_80,code=sm_80 -O3 -lineinfo \
	-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
	-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_OPERATORS__ -D__CUDA_NO_BFLOAT162_OPERATORS__

ifeq ($(VERBOSE), 1)
	CUDAFLAGS += -Xptxas=-v
endif

ifeq ($(PTX), 1)
	CUDAFLAGS += -ptx
endif

# INCLUDES := -I/usr/include/x86_64-linux-gnu/mpi
INCLUDES := 

LIBRARIES := -lcublas -lcublasLt -lcudnn # -lmpi

TARGET_DIR := ./build

TARGETS := $(TARGETS:%=$(TARGET_DIR)/%)

NVCC := nvcc

all: $(TARGETS)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(TARGET_DIR)/%: %.cu common.h | $(TARGET_DIR)
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(LIBRARIES) $< -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu *.h

clean:
	rm -rf $(TARGET_DIR)
