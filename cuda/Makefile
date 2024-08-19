TARGETS := noop fuse memcpy reduce_sum float4 sgemv hgemv sched coalesce transpose loop

CUDAFLAGS := -std=c++17 -gencode arch=compute_70,code=sm_70 -O3 -lineinfo

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
