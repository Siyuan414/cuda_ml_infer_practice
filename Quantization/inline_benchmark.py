"""Tiny throughput-measurement helper, identical math to benchmark_throughput.py
but designed to be called on a model that's already quantized in memory.

Used by the ModelOpt scripts because their HF export format isn't loadable
in gptq_env (it requires modelopt installed at load time). Doing the
measurement inline avoids the round-trip.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import torch

PROMPT = (
    "In recent years, large language models have shown remarkable performance "
    "across a wide range of natural language understanding tasks. The next "
    "frontier is making them efficient enough to deploy at scale, which "
    "requires careful attention to"
)
N_WARMUP = 2
N_RUNS = 5


@torch.no_grad()
def measure_and_dump(
    model,
    tokenizer,
    label: str,
    json_path: Optional[str] = None,
    max_new_tokens: int = 128,
    prompt: str = PROMPT,
    disk_size_mb: Optional[float] = None,
    ppl: Optional[float] = None,
    delta_pct: Optional[float] = None,
) -> dict:
    """Run warm-up + N_RUNS measurement passes, return + optionally dump stats."""
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warm-up
    for _ in range(N_WARMUP):
        _ = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    times = []
    out = None
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    n_input = ids.shape[1]
    n_generated = out.shape[1] - n_input
    mean_t = sum(times) / len(times)
    median_t = sorted(times)[len(times) // 2]
    vram_mb = torch.cuda.memory_allocated() / 1024**2

    payload = {
        "label": label,
        "checkpoint": label,
        "prompt_tokens": n_input,
        "generated_tokens": n_generated,
        "max_new_tokens": max_new_tokens,
        "mean_seconds": mean_t,
        "median_seconds": median_t,
        "tokens_per_sec_mean": n_generated / mean_t,
        "tokens_per_sec_median": n_generated / median_t,
        "all_runs_seconds": times,
        "disk_size_mb": disk_size_mb,
        "vram_allocated_mb": vram_mb,
        "ppl": ppl,
        "delta_pct": delta_pct,
    }

    print(f"\n=== Inline throughput: {label} ===")
    print(f"  prompt tokens          : {n_input}")
    print(f"  generated tokens       : {n_generated}")
    print(f"  median wall time / run : {median_t:.3f}s")
    print(f"  tokens/sec (median)    : {payload['tokens_per_sec_median']:.1f}")
    print(f"  VRAM allocated (MB)    : {vram_mb:.1f}")

    if json_path:
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {json_path}")

    return payload
