TARGETS := noop fuse memcpy reduce_sum add sgemv hgemv sched

CUDAFLAGS := -std=c++17 -gencode arch=compute_70,code=sm_70 -O3 -lcublas

NVCC := nvcc

all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu *.h

clean:
	rm -rf $(TARGETS)
