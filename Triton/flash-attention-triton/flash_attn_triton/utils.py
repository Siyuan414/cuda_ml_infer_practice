"""Benchmarking and correctness helpers."""

import statistics

import torch


def benchmark(fn, *args, iters: int = 50, warmup: int = 10):
    """
    Median GPU time in milliseconds for `fn(*args)`. Uses cuda.Event so CPU-side
    overhead (Python, torch dispatch) is excluded, and a warmup pass so Triton's
    JIT compile time does not contaminate the timing.
    """
    # warmup: triggers the JIT compile on first call, then a few more to settle
    # in any lazy allocations / caches.
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms) / 1000.0  # seconds


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max(torch.abs(a.float() - b.float())).item()
