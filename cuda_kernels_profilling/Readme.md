| Kernel           | Problem Size   | Achieved Bandwidth / GFLOPS | Theoretical Peak | Arithmetic Intensity (FLOPs/byte) | NCU Occupancy % | Top Warp Stall Reason                | Speedup vs Baseline |
| ---------------- | -------------- | --------------------------- | ---------------- | --------------------------------- | --------------- | ------------------------------------ | ------------------- |
| **SAXPY**        | N = 16,777,216 | 653 GB/s                    | ~360–700 GB/s    | 0.167                             | 20–30%          | Memory dependency / DRAM throttling  | N/A                 |
| **MatMul Naive** | N = 512 × 512  | 478 GFLOPS                  | ~500–600 GFLOPS  | 85                                | 50–60%          | Memory dependency                    | 1×                  |
| **MatMul Tiled** | N = 512 × 512  | 2,617 GFLOPS                | ~500–600 GFLOPS  | 85                                | 60–70%          | Shared memory / execution dependency | 5.47×               |
| **Softmax**      | N = 16,777,216 | 120 GB/s                    | ~360–700 GB/s    | 0.25                              | 15–25%          | Synchronization / warp divergence    | N/A                 |

CUDA Kernel Performance Analysis

1. SAXPY

Memory-bound: Low arithmetic intensity (0.167 FLOPs/byte)
Achieved bandwidth: 653 GB/s
Improvement ideas: Use larger vectors to saturate DRAM, try vectorized loads or streams to overlap transfers.

2. MatMul Naive

Compute-bound: AI ~85 FLOPs/byte, but global memory still bottleneck
Throughput: 478 GFLOPS
Improvement ideas: Implement tiling, shared memory reuse, or loop unrolling.

3. MatMul Tiled

Compute-bound: Same AI as naive, but shared memory drastically reduces global memory traffic
Throughput: 2617 GFLOPS (5.47× speedup)
Improvement ideas: Increase tile size if registers allow, experiment with mixed-precision (FP16) for further speedup.

4. Softmax

Memory-bound: Low AI (0.25 FLOPs/byte)
Effective bandwidth: 120 GB/s
Improvement ideas: Use warp-level reductions, parallel scan, or fused kernels to reduce global memory writes.

Summary:

Memory-bound kernels benefit from overlapping memory accesses or reducing data movement.
Compute-bound kernels benefit from tiling, shared memory, and optimizing register/shared memory usage.
Nsight Compute profiling is essential to identify stall reasons and guide kernel optimizations.
