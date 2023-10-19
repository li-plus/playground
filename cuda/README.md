# CUDA Lab

Compile:
```sh
mkdir -p build && cd build
cmake ..
make -j
```

Run benchmark:
```sh
./bin/gemm | grep -v "elapsed"
```

Best results on V100:
```
----- M=128 N=128 K=128 -----
[best] (sgemm3<32, 32, 32, 2, 4, 1, true, false, 1>) vs cublas: 163.8%
----- M=256 N=256 K=256 -----
[best] (sgemm3<32, 32, 32, 2, 4, 1, true, false, 1>) vs cublas: 157.6%
----- M=512 N=512 K=512 -----
[best] (sgemm3<64, 64, 32, 4, 4, 1, true, false, 1>) vs cublas: 94.8%
----- M=1024 N=1024 K=1024 -----
[best] (sgemm3<32, 64, 32, 4, 4, 2, false, false, 8>) vs cublas: 98.2%
----- M=2048 N=2048 K=2048 -----
[best] (sgemm3<64, 64, 64, 4, 4, 8, true, false, 8>) vs cublas: 90.2%
----- M=4096 N=4096 K=4096 -----
[best] (sgemm3<128, 64, 32, 8, 4, 1, true, false, 4>) vs cublas: 96.7%
```

Best results on A100:
```
----- M=128 N=128 K=128 -----
[best] (sgemm3<32, 32, 128, 2, 4, 8, false>) vs cublas: 135.9%
----- M=256 N=256 K=256 -----
[best] (sgemm3<32, 32, 128, 2, 4, 8, true>) vs cublas: 134.2%
----- M=512 N=512 K=512 -----
[best] (sgemm3<64, 64, 64, 4, 4, 8, true>) vs cublas: 87.6%
----- M=1024 N=1024 K=1024 -----
[best] (sgemm3<32, 64, 64, 4, 4, 8, true>) vs cublas: 89.0%
----- M=2048 N=2048 K=2048 -----
[best] (sgemm3<128, 64, 32, 8, 4, 1, true>) vs cublas: 95.6%
----- M=4096 N=4096 K=4096 -----
[best] (sgemm3<64, 64, 32, 8, 4, 2, true>) vs cublas: 94.8%
```

# GPU Arch

A100 GPU white paper: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
