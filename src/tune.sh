#!/usr/bin/env bash

root=$(dirname $(realpath $0))/../
mkdir -p $root/build
cd $root/build
cmake ..

for M in 32 64 128; do
    for N in 32 64 128; do
        for K in 32 64 128; do
            for TY in 1 2 4; do
                echo "M=$M, N=$N, K=$K, TY=$TY"
                sed -e "s/^.*constexpr int BLOCK_SIZE_M.*$/    constexpr int BLOCK_SIZE_M = $M;/g" \
                    -e "s/^.*constexpr int BLOCK_SIZE_N.*$/    constexpr int BLOCK_SIZE_N = $N;/g" \
                    -e "s/^.*constexpr int BLOCK_SIZE_K.*$/    constexpr int BLOCK_SIZE_K = $K;/g" \
                    -e "s/^.*constexpr int THREAD_SIZE_Y.*$/    constexpr int THREAD_SIZE_Y = $TY;/g" \
                    -i ../src/gemm.cu
                make -j >/dev/null 2>&1
                if [[ $? -ne 0 ]]; then continue; fi
                ./bin/gemm_bench
            done
        done
    done
done
