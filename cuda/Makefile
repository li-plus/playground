TARGETS := noop fuse memcpy reduce_sum float4 sgemv hgemv sched coalesce transpose loop

CUDAFLAGS := -std=c++17 -gencode arch=compute_80,code=sm_80 -O3 -lineinfo

ifeq ($(VERBOSE), 1)
	CUDAFLAGS += -Xptxas=-v
endif

LIBRARIES := -lcublas

NVCC := nvcc

all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) $(LIBRARIES) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu *.h

clean:
	rm -rf $(TARGETS)
