# swish(x) = x * sigmoid(x)  — also known as SiLU
#
# This file benchmarks three implementations:
#   1. PyTorch fused  — torch.nn.functional.silu (single CUDA kernel, best baseline)
#   2. PyTorch unfused — manual x * sigmoid(x) (two kernels, extra memory traffic)
#   3. Triton fused   — our own kernel (one kernel, minimal memory traffic)
#
# Expected memory traffic per N elements (float32, 4 bytes each):
#   Fused (ours / SiLU): 2×N×4  — 1 read + 1 write
#   Unfused:             4×N×4  — read x, write sigmoid, read both, write output

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": bs}) for bs in [256, 512, 1024, 2048, 4096]],
    key=["n_elements"],
)
@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * tl.sigmoid(x)   # tl.sigmoid is cleaner than 1 / (1 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def swish_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swish_kernel[grid](x, out, n_elements)
    return out


def swish_unfused(x: torch.Tensor) -> torch.Tensor:
    """Two-kernel baseline: sigmoid and multiply are separate ops."""
    return x * torch.sigmoid(x)


def swish_fused(x: torch.Tensor) -> torch.Tensor:
    """PyTorch's built-in fused SiLU — the fairest PyTorch baseline."""
    return F.silu(x)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(x: torch.Tensor) -> None:
    ref = swish_fused(x)
    triton_out = swish_triton(x)
    max_err = (ref - triton_out).abs().max().item()
    print(f"Max absolute error vs F.silu: {max_err:.2e}")
    assert torch.allclose(ref, triton_out, atol=1e-5), "Correctness check failed!"
    print("Correctness check passed.\n")


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------

def effective_bandwidth_GBps(n_elements: int, elem_bytes: int, n_rw_ops: int, time_s: float) -> float:
    """Bytes moved = n_rw_ops × N × bytes_per_element."""
    return (n_rw_ops * n_elements * elem_bytes) / time_s / 1e9


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    N = 1_000_000
    x = torch.randn(N, device="cuda")

    check_correctness(x)

    # Warmup — ensure kernels are compiled before timing
    for _ in range(10):
        swish_triton(x)
        swish_unfused(x)
        swish_fused(x)

    n_iters = 100
    timer_triton  = benchmark.Timer("swish_triton(x)",  globals={"x": x, "swish_triton":  swish_triton})
    timer_unfused = benchmark.Timer("swish_unfused(x)", globals={"x": x, "swish_unfused": swish_unfused})
    timer_fused   = benchmark.Timer("swish_fused(x)",   globals={"x": x, "swish_fused":   swish_fused})

    res_triton  = timer_triton.timeit(n_iters)
    res_unfused = timer_unfused.timeit(n_iters)
    res_fused   = timer_fused.timeit(n_iters)

    t_triton  = res_triton.mean
    t_unfused = res_unfused.mean
    t_fused   = res_fused.mean

    elem_bytes = x.element_size()  # 4 for float32

    bw_triton  = effective_bandwidth_GBps(N, elem_bytes, n_rw_ops=2, time_s=t_triton)
    bw_unfused = effective_bandwidth_GBps(N, elem_bytes, n_rw_ops=4, time_s=t_unfused)
    bw_fused   = effective_bandwidth_GBps(N, elem_bytes, n_rw_ops=2, time_s=t_fused)

    print(f"{'Implementation':<22} {'Time (µs)':>12} {'Eff. BW (GB/s)':>16} {'Memory ops':>12}")
    print("-" * 66)
    print(f"{'Triton fused':<22} {t_triton  * 1e6:>12.2f} {bw_triton:>16.1f} {'2× N':>12}")
    print(f"{'PyTorch F.silu':<22} {t_fused   * 1e6:>12.2f} {bw_fused:>16.1f} {'2× N':>12}")
    print(f"{'PyTorch unfused':<22} {t_unfused * 1e6:>12.2f} {bw_unfused:>16.1f} {'4× N':>12}")


if __name__ == "__main__":
    main()


#bench mark results 
#Implementation            Time (µs)   Eff. BW (GB/s)   Memory ops
#------------------------------------------------------------------
#Triton fused                  14.33            558.4         2× N
#PyTorch F.silu                 6.10           1312.0         2× N
#PyTorch unfused               21.48            744.7         4× N