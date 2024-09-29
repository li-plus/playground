TARGETS := noop fuse memcpy reduce_sum reduce_sum_dim float4 sgemv hgemv sched coalesce transpose loop cache ipc_ag softmax gemm_bias_gelu vote sort conv sm_sched

CUDAFLAGS := -std=c++17 -gencode arch=compute_80,code=sm_80 -O3 -lineinfo

ifeq ($(VERBOSE), 1)
	CUDAFLAGS += -Xptxas=-v
endif

INCLUDES := -I/usr/include/x86_64-linux-gnu/mpi

LIBRARIES := -lcublas -lcublasLt -lcudnn -lmpi

TARGET_DIR := ./build

TARGETS := $(TARGETS:%=$(TARGET_DIR)/%)

NVCC := nvcc

all: $(TARGETS)

$(TARGET_DIR)/%: %.cu common.h
	mkdir -p $(TARGET_DIR)
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) $(LIBRARIES) $< -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu *.h

clean:
	rm -rf $(TARGET_DIR)
