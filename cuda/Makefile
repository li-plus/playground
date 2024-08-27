TARGETS := noop fuse memcpy reduce_sum float4 sgemv hgemv sched coalesce transpose loop cache

CUDAFLAGS := -std=c++17 -gencode arch=compute_80,code=sm_80 -O3 -lineinfo

ifeq ($(VERBOSE), 1)
	CUDAFLAGS += -Xptxas=-v
endif

LIBRARIES := -lcublas

TARGET_DIR := ./bin

TARGETS := $(TARGETS:%=$(TARGET_DIR)/%)

NVCC := nvcc

all: $(TARGETS)

$(TARGET_DIR)/%: %.cu
	mkdir -p $(TARGET_DIR)
	$(NVCC) $(CUDAFLAGS) $(LIBRARIES) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu *.h

clean:
	rm -rf $(TARGET_DIR)
