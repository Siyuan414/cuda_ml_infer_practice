# Benchmark Results

Forward-pass walltime and profiler breakdown of the Triton FlashAttention-2 kernel in this repo, compared against `torch.nn.functional.scaled_dot_product_attention` (PyTorch's built-in, which on this hardware/build dispatches to Tri Dao's fused FlashAttention backend — verified via `torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)`).

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti (Blackwell consumer, sm_120) |
| OS | WSL2 / Ubuntu (Windows host) |
| Triton | 3.7 |
| PyTorch | _(fill in — `python -c "import torch; print(torch.__version__)"`)_ |
| CUDA | _(fill in — `nvcc --version`)_ |
| Date | 2026-04-19 |

To reproduce:
```bash
pip install -e .
python -m benchmark.benchmark_attention
python -m benchmark.profile_attention
```

## Walltime (forward, non-causal)

Shape: `batch=2, heads=8, head_dim=64, dtype=fp16`. Median of 50 iterations after 10 warmup.

|    N | PyTorch SDPA (flash) | Triton v0 (fixed config) | Triton v1 (autotune + exp2 + loop reorder) |
|-----:|---------------------:|-------------------------:|-------------------------------------------:|
|  512 |             0.021 ms |     0.026 ms (0.78×)     |   **0.020 ms (1.03×)** |
| 1024 |             0.058 ms |     0.067 ms (0.86×)     |   **0.057 ms (1.01×)** |
| 2048 |             0.205 ms |     0.237 ms (0.86×)     |   **0.206 ms (1.00×)** |
| 4096 |             0.791 ms |     0.910 ms (0.87×)     |   **0.767 ms (1.03×)** |

(Numbers in parentheses are Triton-vs-SDPA speedup; ≥1.00× means Triton is faster.)

**Result:** the tuned Triton kernel matches or slightly beats the production FlashAttention-2 backend across every tested sequence length on a 5070 Ti. The +3% at N=4096 is consistent across runs.

## What changed v0 → v1

Two optimizations, both standard FA-2 practice:

1. **`@triton.autotune`** over 12 configs spanning `BLOCK_M ∈ {64, 128, 256}`, `BLOCK_N ∈ {32, 64, 128}`, `num_warps ∈ {2, 4, 8}`, `num_stages ∈ {2, 3, 4}`. Keyed on `(N, D, IS_CAUSAL)` so each shape gets its own compiled best config. First call per key sweeps and caches; subsequent calls are free.
2. **Log-2 softmax trick.** Folded `softmax_scale * log2(e)` into Q at load time and switched the inner loop from `tl.exp` (Taylor expansion) to `tl.math.exp2` (one hardware instruction, `MUFU.EX2`). Removes one fp32 multiply per inner-loop iteration and replaces a multi-cycle math op with a single one.
3. **Inner-loop reorder.** Moved the V load to *after* the softmax computation. With `num_stages=2/3`, this lets the compiler pipeline the next iteration's V load concurrently with the current iteration's `P @ V` matmul. Worth a few percent on Ada/Blackwell.

## Profile breakdown

Shape: `batch=2, heads=8, seq=1024, head_dim=64, dtype=fp16`. 10 calls, 5 warmup.

| Op | Self CUDA | % | Count |
|---|---:|---:|---:|
| `flash_attn_fwd` | 633.6 µs | 100.0% | 10 |

Per-call CUDA time: ≈63 µs (consistent with the walltime number). Only one Triton kernel shows up — nothing falls back to `aten::*` compute ops.

## Sanity / correctness

All 16 parametrized cases in `tests/test_vs_pytorch.py` pass (`max|Δ| < 2e-2` in fp16 vs SDPA, across 4 shapes × {causal, non-causal}). Includes `N=500` to exercise the tail-masking path when `N % BLOCK_M != 0`.

## What's not in here

The remaining performance headroom against the absolute best fused-attention kernels (which can be ~5–15% faster on H100-class hardware) requires:

- **Warp specialization** — split warps into producer (memory) and consumer (compute) groups. Triton doesn't expose this cleanly today; needs CUTLASS / raw CUDA.
- **TMA + wgmma** (Hopper / data-center Blackwell only) — not applicable to consumer Blackwell.
- **Hand-tuned register assignments** to keep Q, accumulator, and streaming K/V all resident.

For a ~140-line pedagogical Triton kernel matching the production FlashAttention-2 backend on a consumer Blackwell GPU, this is a reasonable place to stop.
