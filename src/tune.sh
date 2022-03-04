#!/usr/bin/env bash

root=$(dirname $(realpath $0))/../
mkdir -p $root/build
cd $root/build
cmake ..

for BM in 32 64 128; do
    for BN in 32 64 128; do
        for BK in 32 64 128; do
            for TY in 1 2 4; do
                for PREFETCH in "false" "true"; do
                    echo "BM=$BM, BN=$BN, BK=$BK, TY=$TY, PREFETCH=$PREFETCH"
                    sed -e "s/^.*constexpr int BLOCK_SIZE_M.*$/    constexpr int BLOCK_SIZE_M = $BM;/g" \
                        -e "s/^.*constexpr int BLOCK_SIZE_N.*$/    constexpr int BLOCK_SIZE_N = $BN;/g" \
                        -e "s/^.*constexpr int BLOCK_SIZE_K.*$/    constexpr int BLOCK_SIZE_K = $BK;/g" \
                        -e "s/^.*constexpr int THREAD_SIZE_Y.*$/    constexpr int THREAD_SIZE_Y = $TY;/g" \
                        -e "s/^.*constexpr bool DO_PREFETCH.*$/    constexpr bool DO_PREFETCH = $PREFETCH;/g" \
                        -i ../src/gemm.cu
                    make -j >/dev/null 2>&1 && ./bin/gemm_bench
                done
            done
        done
    done
done
