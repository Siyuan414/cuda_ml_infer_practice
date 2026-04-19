"""Walltime comparison: Triton FlashAttention-2 forward vs torch.nn.functional.SDPA."""

# Allow `python benchmark/benchmark_attention.py` from the repo root without
# `pip install -e .`. When invoked as a module (`python -m benchmark.…`) or
# after a proper install, this is a no-op.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from flash_attn_triton.utils import benchmark
from flash_attn_triton.wrapper import flash_attention


def run():
    B, H, D = 2, 8, 64
    print(f"batch={B}  heads={H}  head_dim={D}  dtype=fp16  (median of 50, 10 warmup)")
    print(f"{'N':>6}  {'PyTorch (ms)':>14}  {'Triton (ms)':>13}  {'speedup':>8}")

    for N in [512, 1024, 2048, 4096]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        t_pt = benchmark(lambda: F.scaled_dot_product_attention(q, k, v))
        t_tr = benchmark(lambda: flash_attention(q, k, v))

        speedup = t_pt / t_tr if t_tr > 0 else float("inf")
        print(f"{N:>6}  {t_pt * 1000:>14.3f}  {t_tr * 1000:>13.3f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    run()
