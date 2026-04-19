"""Python entry point that launches the Triton kernel."""

from __future__ import annotations

import math

import torch
import triton

from .kernel import flash_attn_fwd


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """
    FlashAttention-2 forward pass.

    Args:
        q, k, v: (B, H, N, D), fp16 or bf16, contiguous.
        causal:  whether to apply a causal mask.
        softmax_scale: defaults to 1/sqrt(D).

    Returns:
        out: (B, H, N, D), same dtype as q.

    Notes:
        Block sizes, num_warps, and num_stages are chosen by `@triton.autotune`.
        The FIRST call for a given (N, D, causal) combination runs the autotune
        sweep and may take several seconds; subsequent calls use the cached best
        config. Use `warmup > 0` in any benchmarking.
    """
    assert q.shape == k.shape == v.shape, "Q, K, V must share shape"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "all inputs must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16), \
        "use fp16 or bf16 so tl.dot dispatches to tensor cores"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    B, H, N, D = q.shape
    assert D in {16, 32, 64, 128, 256}, f"head_dim must be a power of two in 16..256, got {D}"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    o = torch.empty_like(q)

    # With autotune, BLOCK_M/BLOCK_N/num_warps/num_stages come from the chosen
    # config. The grid is a callable so it can read the selected BLOCK_M.
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)

    flash_attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H, N,
        softmax_scale,
        D=D,
        IS_CAUSAL=causal,
    )
    return o
