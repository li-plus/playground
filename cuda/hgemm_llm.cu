#include "common.h"

// 2. store to shared memory before global memory
// 3. __launch_bounds__
// 4. common var before loop
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Wrapper for ldmatrix A (loads 4 uint32_t registers)
__device__ __forceinline__ uint4 ldmatrix_x4_m8n8(const half* addr) {
    uint32_t saddr = __cvta_generic_to_shared(addr);
    uint4 result;
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "r"(saddr)
    );
    return result;
}

// Wrapper for ldmatrix B with transpose x4 (loads 4 uint32_t registers for x4)
__device__ __forceinline__ uint4 ldmatrix_x4_m8n8_trans(const half* addr) {
    uint32_t saddr = __cvta_generic_to_shared(addr);
    uint4 result;
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
        : "r"(saddr)
    );
    return result;
}

// Wrapper for MMA m16n8k16
__device__ __forceinline__ uint2 mma_m16n8k16(uint4 a, uint2 b, uint2 c) {
    uint2 result;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};"
        : "=r"(result.x), "=r"(result.y)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
          "r"(b.x), "r"(b.y),
          "r"(c.x), "r"(c.y)
    );
    return result;
}

// Wrapper for cp.async 16-byte transfer with optimized cache policy
__device__ __forceinline__ void cp_async_128(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

// Commit and wait for cp.async operations
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group_2() {
    asm volatile("cp.async.wait_group 2;\n" ::: "memory");
}

// Swizzle functions to minimize bank conflicts
// Group 8 adjacent elements as a unit (16 bytes = cp.async transfer size)
__device__ __forceinline__ int swizzle_A(int linear_offset) {
    // A matrix: 128x32, but swizzle with stride 64 to fit 32 4-byte banks
    // Group 8 elements as unit, stride = 64, so 64/8 = 8 units per swizzle row
    int logical_row = linear_offset / 64;           // Logical row based on 64-element stride
    int element_in_swizzle_row = linear_offset % 64; // Element within swizzle row
    int logical_unit_col = element_in_swizzle_row / 8; // Unit column (0-7)
    int element_in_unit = element_in_swizzle_row % 8;  // Element within unit (0-7)
    
    int physical_unit_col = logical_unit_col ^ (logical_row % 8);  // XOR with row mod 8
    physical_unit_col = physical_unit_col % 8;      // Ensure within range [0,7]
    
    int physical_offset = logical_row * 64 + physical_unit_col * 8 + element_in_unit;
    return physical_offset;
}

__device__ __forceinline__ int swizzle_B(int linear_offset) {
    // B matrix: 32x256, group 8 elements as unit, width = 256/8 = 32 units per row
    int logical_row = linear_offset / 256;          // Logical row
    int element_in_row = linear_offset % 256;       // Element within row
    int logical_unit_col = element_in_row / 8;      // Unit column (0-31)
    int element_in_unit = element_in_row % 8;       // Element within unit (0-7)
    
    int physical_unit_col = logical_unit_col ^ (logical_row % 32);  // XOR with row mod 32
    physical_unit_col = physical_unit_col % 32;     // Ensure within range [0-31]
    
    int physical_offset = logical_row * 256 + physical_unit_col * 8 + element_in_unit;
    return physical_offset;
}

__global__ void hgemm_kernel(const half *A, const half *B, half *C, int M, int N, int K) {
    // Dynamic shared memory allocation for 4-stage prefetching
    // A: 4 * 128 * 32, B: 4 * 32 * 256
    // Total: 4 * (128*32 + 32*256) * sizeof(half) = 4 * (4096 + 8192) * 2 = 98304 bytes = 96KB
    extern __shared__ half smem[];
    
    // Convert 1D shared memory to 2D arrays
    half (*sA)[128 * 32] = (half(*)[128 * 32])smem;
    half (*sB)[32 * 256] = (half(*)[32 * 256])(smem + 4 * 128 * 32);
    
    int tid = threadIdx.x;  // Thread ID within block (0-255 for 8 warps)
    int warp_id = tid / 32; // Warp ID within block (0-7)
    int lane_id = tid % 32; // Lane ID within warp (0-31)
    
    // Implement thread block swizzling with zig-zag pattern (size 8)
    // Convert linear block ID to swizzled block coordinates
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int swizzle_group_id = linear_block_id / 64;  // 64 blocks per group (8x8)
    int block_in_group = linear_block_id % 64;    // Position within group
    
    int swizzle_row = block_in_group / 8;         // Row within 8x8 group (0-7)
    int swizzle_col = block_in_group % 8;         // Col within 8x8 group (0-7)
    
    // Calculate final swizzled block coordinates
    int blocks_per_row = gridDim.x / 8;           // Number of 8x8 groups per row
    int group_row = swizzle_group_id / blocks_per_row;  // Which row of groups
    int group_col = swizzle_group_id % blocks_per_row;  // Which col of groups
    
    int block_row = group_row * 8 + swizzle_row;  // Final swizzled block row
    int block_col = group_col * 8 + swizzle_col;  // Final swizzled block col
    
    // Calculate warp position within the block (2x4 warp grid)
    int warp_row = warp_id / 4;  // Warp row (0-1) within 2x4 warp grid
    int warp_col = warp_id % 4;  // Warp col (0-3) within 2x4 warp grid
    
    // Calculate global C matrix position for this warp (64x64 per warp)
    int C_row_start = block_row * 128 + warp_row * 64;
    int C_col_start = block_col * 256 + warp_col * 64;
    
    // Initialize accumulators for 32 MMA operations using default initialization
    uint2 rC[32] = {};  // Default initialize to zero
    
    // Helper lambda to load A and B blocks for a given k_block and stage
    auto load_GEMM_tile = [&](int k_block, int stage) {
        // Load A block (128x32) into shared memory using cp.async with swizzling
        // 256 threads, each loads 16 elements (2 float4), 2 iterations total
        #pragma unroll
        for (int load_iter = 0; load_iter < 2; load_iter++) {
            int A_row = block_row * 128 + (tid / 4) + load_iter * 64;  // 64 rows per iteration, 4 threads per row
            int A_col = k_block + (tid % 4) * 8;  // Each thread loads 8 consecutive elements
            int A_idx = A_row * K + A_col;
            
            int sA_linear = ((tid / 4) + load_iter * 64) * 32 + (tid % 4) * 8;  // Linear shared memory offset
            int sA_swizzled = swizzle_A(sA_linear);  // Apply swizzle to linear offset
            
            // Use cp.async to load 16 bytes (8 half values) asynchronously with swizzling
            cp_async_128(&sA[stage][sA_swizzled], &A[A_idx]);
        }
        
        // Load B block (32x256) into shared memory using cp.async with swizzling
        // 256 threads, each loads 32 elements (4 float4), 4 iterations total
        #pragma unroll
        for (int load_iter = 0; load_iter < 4; load_iter++) {
            int B_row = k_block + (tid / 32) + load_iter * 8;  // 8 rows per iteration, 32 threads per row
            int B_col = block_col * 256 + (tid % 32) * 8;  // Each thread loads 8 consecutive elements
            int B_idx = B_row * N + B_col;
            
            int sB_linear = ((tid / 32) + load_iter * 8) * 256 + (tid % 32) * 8;  // Linear shared memory offset
            int sB_swizzled = swizzle_B(sB_linear);  // Apply swizzle to linear offset
            
            // Use cp.async to load 16 bytes (8 half values) asynchronously with swizzling
            cp_async_128(&sB[stage][sB_swizzled], &B[B_idx]);
        }
        
        cp_async_commit_group();
    };
    
    // Helper lambda to perform MMA operations for a given stage
    auto compute_GEMM_tile = [&](int stage, bool wait_and_prefetch = false, int next_stage = -1) {
        // Double buffer for A and B fragments (2x fragment registers)
        uint4 rA_fragments[2][4];  // [buffer][tile_m] - 2 buffers for A fragments
        uint4 rB_fragments[2][4];  // [buffer][tile_n] - 2 buffers for B fragments
        
        // Pre-load fragments for k_sub=0 into buffer 0
        int k_offset = 0;
        #pragma unroll
        for (int tile_m = 0; tile_m < 4; tile_m++) {
            int A_tile_row = warp_row * 64 + tile_m * 16;
            int lane_group = lane_id / 16;
            int lane_in_group = lane_id % 16;
            
            int sA_linear = (A_tile_row + lane_in_group) * 32 + k_offset + lane_group * 8;
            int sA_swizzled = swizzle_A(sA_linear);
            const half* sA_addr = &sA[stage][sA_swizzled];
            
            rA_fragments[0][tile_m] = ldmatrix_x4_m8n8(sA_addr);
        }
        
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; tile_n++) {
            int B_tile_col = warp_col * 64 + tile_n * 16;
            int lane_group = lane_id / 16;
            int lane_in_group = lane_id % 16;
            
            int sB_linear = (k_offset + lane_in_group) * 256 + B_tile_col + lane_group * 8;
            int sB_swizzled = swizzle_B(sB_linear);
            const half* sB_addr = &sB[stage][sB_swizzled];
            
            rB_fragments[0][tile_n] = ldmatrix_x4_m8n8_trans(sB_addr);
        }
        
        // Process both K=16 slices within the K=32 block
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; k_sub++) {
            int current_buffer = k_sub % 2;
            int next_buffer = (k_sub + 1) % 2;
            
            // Prefetch logic
            if (k_sub == 0) {
                // Prefetch k_sub=1 from current stage
                int next_k_offset = 16;
                #pragma unroll
                for (int tile_m = 0; tile_m < 4; tile_m++) {
                    int A_tile_row = warp_row * 64 + tile_m * 16;
                    int lane_group = lane_id / 16;
                    int lane_in_group = lane_id % 16;
                    
                    int sA_linear = (A_tile_row + lane_in_group) * 32 + next_k_offset + lane_group * 8;
                    int sA_swizzled = swizzle_A(sA_linear);
                    const half* sA_addr = &sA[stage][sA_swizzled];
                    
                    rA_fragments[next_buffer][tile_m] = ldmatrix_x4_m8n8(sA_addr);
                }
                
                #pragma unroll
                for (int tile_n = 0; tile_n < 4; tile_n++) {
                    int B_tile_col = warp_col * 64 + tile_n * 16;
                    int lane_group = lane_id / 16;
                    int lane_in_group = lane_id % 16;
                    
                    int sB_linear = (next_k_offset + lane_in_group) * 256 + B_tile_col + lane_group * 8;
                    int sB_swizzled = swizzle_B(sB_linear);
                    const half* sB_addr = &sB[stage][sB_swizzled];
                    
                    rB_fragments[next_buffer][tile_n] = ldmatrix_x4_m8n8_trans(sB_addr);
                }
            } else if (wait_and_prefetch && next_stage >= 0) {
                // For k_sub=1: wait for next cp.async and prefetch k_sub=0 from next stage
                cp_async_wait_group_2();
                __syncthreads();
                
                int next_k_offset = 0;  // k_sub=0 from next stage
                #pragma unroll
                for (int tile_m = 0; tile_m < 4; tile_m++) {
                    int A_tile_row = warp_row * 64 + tile_m * 16;
                    int lane_group = lane_id / 16;
                    int lane_in_group = lane_id % 16;
                    
                    int sA_linear = (A_tile_row + lane_in_group) * 32 + next_k_offset + lane_group * 8;
                    int sA_swizzled = swizzle_A(sA_linear);
                    const half* sA_addr = &sA[next_stage][sA_swizzled];
                    
                    rA_fragments[0][tile_m] = ldmatrix_x4_m8n8(sA_addr);  // Load into buffer 0 for next iteration
                }
                
                #pragma unroll
                for (int tile_n = 0; tile_n < 4; tile_n++) {
                    int B_tile_col = warp_col * 64 + tile_n * 16;
                    int lane_group = lane_id / 16;
                    int lane_in_group = lane_id % 16;
                    
                    int sB_linear = (next_k_offset + lane_in_group) * 256 + B_tile_col + lane_group * 8;
                    int sB_swizzled = swizzle_B(sB_linear);
                    const half* sB_addr = &sB[next_stage][sB_swizzled];
                    
                    rB_fragments[0][tile_n] = ldmatrix_x4_m8n8_trans(sB_addr);  // Load into buffer 0 for next iteration
                }
            }
            
            // Perform all MMA operations using current buffer
            int mma_idx = 0;
            #pragma unroll
            for (int tile_m = 0; tile_m < 4; tile_m++) {
                #pragma unroll
                for (int tile_n = 0; tile_n < 4; tile_n++) {
                    uint4 rA = rA_fragments[current_buffer][tile_m];
                    uint4 rB_full = rB_fragments[current_buffer][tile_n];
                    
                    uint2 rB_left = make_uint2(rB_full.x, rB_full.y);
                    uint2 rB_right = make_uint2(rB_full.z, rB_full.w);
                    
                    rC[mma_idx] = mma_m16n8k16(rA, rB_left, rC[mma_idx]);
                    rC[mma_idx + 1] = mma_m16n8k16(rA, rB_right, rC[mma_idx + 1]);
                    
                    mma_idx += 2;
                }
            }
        }
    };
    
    // Prefetch first 3 stages (k=0, k=1, k=2)
    #pragma unroll
    for (int prefetch_k = 0; prefetch_k < 3; prefetch_k++) {
        int k_block = prefetch_k * 32;
        if (k_block < K) {
            load_GEMM_tile(k_block, prefetch_k % 4);
        }
    }
    
    // Main loop starting from k=3, overlapping computation and memory access
    for (int k = 3; k * 32 < K; k++) {
        int k_block = k * 32;
        int compute_stage = (k - 3) % 4;  // Stage to compute (k-3)
        int load_stage = k % 4;           // Stage to load current k
        
        // Issue cp.async for current k (no wait needed - handled in compute_GEMM_tile)
        if (k_block < K) {
            load_GEMM_tile(k_block, load_stage);
        }
        
        // Perform MMA for k-3 with cross-stage prefetching
        int next_compute_stage = (k - 2) % 4;  // Stage for k-2 (next iteration)
        compute_GEMM_tile(compute_stage, true, next_compute_stage);
    }
    
    // Finish remaining 3 MMA operations
    cp_async_wait_group_0();  // Wait for all remaining cp.async operations
    __syncthreads();
    
    #pragma unroll
    for (int remaining = 0; remaining < 3; remaining++) {
        int k = (K / 32) - 3 + remaining;  // Calculate the k index for remaining stages
        if (k >= 0 && k * 32 < K) {
            int compute_stage = k % 4;
            bool is_last_two = (remaining >= 1);  // Last 2 iterations don't prefetch
            int next_stage = is_last_two ? -1 : ((k + 1) % 4);
            compute_GEMM_tile(compute_stage, !is_last_two, next_stage);
        }
    }
    
    // Store results back to global memory using half2 for adjacent elements
    int mma_idx = 0;
    
    #pragma unroll
    for (int tile_m = 0; tile_m < 4; tile_m++) {
        #pragma unroll
        for (int tile_n = 0; tile_n < 4; tile_n++) {
            // Calculate base position for this tile
            int tile_row_start = C_row_start + tile_m * 16;
            int tile_col_start = C_col_start + tile_n * 16;
            
            int quad = lane_id / 4;
            int lane_in_quad = lane_id % 4;
            int base_row = tile_row_start + quad;
            
            // Store left 16x8 fragment using half2
            half* output_ptr_left = (half*)&rC[mma_idx];
            int base_col_left = tile_col_start + lane_in_quad * 2;
            
            half2* C_ptr_left_0 = (half2*)&C[base_row * N + base_col_left];
            half2* C_ptr_left_8 = (half2*)&C[(base_row + 8) * N + base_col_left];
            
            *C_ptr_left_0 = *((half2*)&output_ptr_left[0]);  // Store elements 0,1
            *C_ptr_left_8 = *((half2*)&output_ptr_left[2]);  // Store elements 2,3
            
            // Store right 16x8 fragment using half2
            half* output_ptr_right = (half*)&rC[mma_idx + 1];
            int base_col_right = tile_col_start + 8 + lane_in_quad * 2;
            
            half2* C_ptr_right_0 = (half2*)&C[base_row * N + base_col_right];
            half2* C_ptr_right_8 = (half2*)&C[(base_row + 8) * N + base_col_right];
            
            *C_ptr_right_0 = *((half2*)&output_ptr_right[0]);  // Store elements 0,1
            *C_ptr_right_8 = *((half2*)&output_ptr_right[2]);  // Store elements 2,3
            
            mma_idx += 2;
        }
    }
}

void hgemm(const half *A, const half *B, half *C, int M, int N, int K) {
    // Calculate required shared memory size for 4-stage prefetching
    // A: 4 * 128 * 32, B: 4 * 32 * 256
    // Total: 4 * (128*32 + 32*256) * sizeof(half) = 4 * (4096 + 8192) * 2 = 98304 bytes = 96KB
    const int smem_size = 4 * (128 * 32 + 32 * 256) * sizeof(half);  // 96KB
    
    // Set maximum shared memory size for the kernel
    cudaFuncSetAttribute(hgemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    // Each block contains 2x4=8 warps and computes a 128x256 tile of C
    dim3 blockDim(256);  // 8 warps per block (32 threads * 8 warps)
    dim3 gridDim(N / 256, M / 128);  // Grid covers entire C matrix
    
    // Launch kernel with dynamic shared memory
    hgemm_kernel<<<gridDim, blockDim, smem_size>>>(A, B, C, M, N, K);
}