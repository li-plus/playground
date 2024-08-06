TARGETS := noop fuse memcpy reduce_sum add gemv

CUDAFLAGS := -std=c++17 -gencode arch=compute_70,code=sm_70 -O3 -lcublas

NVCC := nvcc

all: $(TARGETS)

%: %.cu
	$(NVCC) $(CUDAFLAGS) $^ -o $@

.PHONY: lint clean

lint:
	clang-format -i *.cu

clean:
	rm -rf $(TARGETS)
