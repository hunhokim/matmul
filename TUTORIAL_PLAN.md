# CUDA Matmul Tutorial Plan

Implement `C[M,N] = A[M,K] × B[K,N]` with row-major FP32 matrices, including rectangular shapes and partial tiles. Keep numbered kernels in `src/` with a shared correctness and benchmark harness. Change one thing per step.

| Step | Change | Main lesson |
| --- | --- | --- |
| 0 | Build CPU reference, CUDA error checks, and event timing. | Trust correctness and measurements. |
| 1 | One thread computes one output; loop over `K`. | Indexing, registers, input loads. |
| 2 | Compare threads spanning columns versus rows; retain the better mapping. | Warp access patterns and coalescing. |
| 3 | Cooperatively stage A tiles in shared memory. | Reuse across threads and barriers. |
| 4 | Stage B tiles too: load → synchronize → compute → synchronize. | Shared-memory tiling: one output tile per block. |
| 5 | Each thread computes a column of outputs. | Register tiling; reuse B across accumulators. |
| 6 | Each thread computes a rectangle of outputs. | Reuse both operands; balance register pressure. |
| 7 | Tune block tile, reduction chunk, and thread tile dimensions individually. | Reuse versus occupancy and available parallelism. |

**Milestones:** Complete steps 0–4 first, then 5–7. Shared-memory tiling adds cooperative input reuse; register tiling lets a block compute more outputs without adding threads. Speedups are hypotheses to test.

## Optional extensions

Choose based on profiling and hardware support; apply separately.

| Change | Investigate |
| --- | --- |
| Shared-memory padding/layout | Bank conflicts. |
| Loop unrolling | Instruction overhead versus register use. |
| Vectorized loads | Alignment, tails, and load instruction count. |
| Warp tiles | Ownership and shared-memory access patterns. |
| Double buffering | Safe buffer reuse. |
| Asynchronous copies | Overlap loading with computation. |
| Multiple successive output tiles per block | Scheduling/reuse benefits versus reduced parallelism. |
| Tensor Cores, in a separate branch | Collective operations and precision tradeoffs. |

## For every step

- **Reason:** Draw output ownership; trace one warp's addresses; count reuse, barriers, registers, and shared memory.
- **Validate:** Compare with a double-accumulated CPU reference using absolute/relative tolerances. Test tiny, rectangular, and partial-tile cases, e.g. `(127,259,65)`.
- **Synchronize safely:** Zero-fill invalid input loads, guard stores, and keep all threads participating in block barriers. Run memory/synchronization checks when adding shared memory or pipelines.
- **Measure:** Warm up; report median kernel time, `2MNK/time` FLOP/s, and speedup across several sizes. Exclude allocation/transfers; record GPU and launch configuration.
- **Explain:** Record the prediction, result, and likely bottleneck. Compare against cuBLAS with matching layout and precision.

References: [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab) · [CUTLASS tiling](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
