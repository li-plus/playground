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
[best] (sgemm3<32, 32, 64, 4, 2, 4, true, true, 1>) vs cublas: 173.8% (0.005 vs 0.008 ms)
----- M=256 N=256 K=256 -----
[best] (sgemm3<32, 32, 64, 4, 2, 4, true, true, 1>) vs cublas: 117.2% (0.007 vs 0.009 ms)
----- M=512 N=512 K=512 -----
[best] (sgemm3<128, 32, 32, 8, 2, 4, true, true, 1>) vs cublas: 93.8% (0.030 vs 0.028 ms)
----- M=1024 N=1024 K=1024 -----
[best] (sgemm3<128, 64, 32, 8, 4, 4, true, true, 1>) vs cublas: 98.0% (0.186 vs 0.182 ms)
----- M=2048 N=2048 K=2048 -----
[best] (sgemm3<128, 64, 32, 8, 4, 4, true, true, 1>) vs cublas: 92.2% (1.277 vs 1.178 ms)
----- M=4096 N=4096 K=4096 -----
[best] (sgemm3<128, 64, 32, 8, 4, 4, true, true, 1>) vs cublas: 98.9% (9.419 vs 9.319 ms)
```

Best results on A100:
```
----- M=128 N=128 K=128 -----
[best] (sgemm3<32, 32, 128, 2, 4, 8, false, false, 1>) vs cublas: 131.7%
----- M=256 N=256 K=256 -----
[best] (sgemm3<32, 32, 128, 2, 4, 8, true, false, 1>) vs cublas: 134.1%
----- M=512 N=512 K=512 -----
[best] (sgemm3<64, 64, 64, 4, 4, 8, true, false, 4>) vs cublas: 86.6%
----- M=1024 N=1024 K=1024 -----
[best] (sgemm3<32, 64, 64, 4, 4, 8, true, false, 1>) vs cublas: 88.9%
----- M=2048 N=2048 K=2048 -----
[best] (sgemm3<128, 64, 32, 8, 4, 1, true, false, 1>) vs cublas: 95.5%
----- M=4096 N=4096 K=4096 -----
[best] (sgemm3<64, 64, 32, 8, 4, 2, true, false, 1>) vs cublas: 94.9%
```

# GPU Arch

A100 GPU white paper: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

# Nsight System

References:
* https://docs.nvidia.com/nsight-systems/UserGuide/index.html
* https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223

Profile command:
```sh
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsys_report -f true -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --cudabacktrace=kernel --python-backtrace=cuda --python-sampling=true \
    python3 multi_stream.py
```

To capture activities within specific NVTX range only, add these options:
```sh
-c nvtx -p forward
```

# Nsight Compute

Download from https://developer.nvidia.com/tools-overview/nsight-compute/get-started.

Install Nsight Compute GUI on host. Should be an interactive installation guide.

Install Nsight Compute CLI on server:
```sh
bash nsight-compute-linux-2024.3.0.15-34567288.run
```

Profile kernel `memcpy_cuda_kernel` on server. Collect results of 4 kernels after skipping 2 kernels.
```sh
sudo -E $(which ncu) --set detailed -o profile -f -k memcpy_cuda_kernel -s 2 -c 4 ./memcpy
```

To analyze memory (e.g. bank conflicts), add options:
```sh
--section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables
```

See https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html for usage details.

# SM Occupancy

https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm

Use Nsight Compute to calculate theoretical max occupancy.

# TensorRT-LLM

Follow instructions of the [LLaMA example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama).

For half precision, cublas gemm kernel is used for context. [cudaCoreGemm](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/cudaCoreGemm.cu) kernel is used to compute GEMV for decoding.

For `tp_size > 1`, [oneShotAllReduceKernel](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu) is used for communication by default. Also supports NCCL strategy.

For weight only quantization, `GemmFpAIntB` is used for context. Custom [gemv kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/kernel.h) is used for decoing.

Sadly, batch manager & executor modules are not open source.
