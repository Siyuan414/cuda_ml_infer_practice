"""FlashAttention-2 forward kernel (Triton), autotuned.

Shape contract:
    Q, K, V, O : (B, H, N, D)  fp16 or bf16, contiguous on the last dim
    D must be a power of two in {16, 32, 64, 128, 256}

Grid:
    axis 0 = cdiv(N, BLOCK_M)   -> which BLOCK_M-tile of queries this program owns
    axis 1 = B * H              -> which (batch, head) pair this program owns
"""

import triton
import triton.language as tl


# Autotune sweep tuned for consumer Blackwell / Ada Lovelace (sm_120 / sm_89):
#   - 48 KB shared memory per SM, so deep pipelines (num_stages=3+) can run out
#     of room for D=128/256 — keep some num_stages=2 entries.
#   - Smaller tiles (BLOCK_M=64) help when the grid would otherwise be too small
#     to saturate the SMs (e.g. short sequences with small batch/heads).
#   - Larger tiles (BLOCK_M=256) reduce loop overhead at long N where launch
#     parallelism is no longer the bottleneck.
# Configs that don't fit (too much SRAM for the chosen D) are silently skipped
# by Triton at compile time, so it's safe to leave a few aggressive ones in.
_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=4),
    # smaller tiles for low-parallelism shapes
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=2, num_stages=3),
    # larger tiles for very long N
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 32},  num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["N", "D", "IS_CAUSAL"])
@triton.jit
def flash_attn_fwd(
    Q, K, V, O,
    # batch + head strides (in elements) for each tensor
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # runtime shape params
    N_HEADS,
    N,
    softmax_scale,
    # compile-time constants
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # ── program coordinates ──
    pid_m = tl.program_id(0)                # which BLOCK_M tile of queries
    pid_bh = tl.program_id(1)               # flat (batch, head) index
    off_b = pid_bh // N_HEADS
    off_h = pid_bh %  N_HEADS

    # ── shift base pointers to the current (batch, head) ──
    Q_bh = Q + off_b * stride_qb + off_h * stride_qh
    K_bh = K + off_b * stride_kb + off_h * stride_kh
    V_bh = V + off_b * stride_vb + off_h * stride_vh
    O_bh = O + off_b * stride_ob + off_h * stride_oh

    # ── index vectors ──
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # ── log-2 softmax trick: fold softmax_scale * log2(e) into Q at load time ──
    # so the inner loop uses tl.math.exp2 (one HW instruction) instead of tl.exp.
    LOG2E: tl.constexpr = 1.4426950408889634
    qk_scale = softmax_scale * LOG2E

    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)
    q = (q * qk_scale).to(q.dtype)          # keep Q in fp16/bf16 for tl.dot

    # ── online-softmax running stats, fp32 ──
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    if IS_CAUSAL:
        hi = (pid_m + 1) * BLOCK_M
    else:
        hi = N

    for start_n in range(0, hi, BLOCK_N):
        # ── load K^T tile ──
        k_ptrs = K_bh + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < N, other=0.0)

        # ── Q·K^T (already base-2-scaled because of the qk_scale fold above) ──
        scores = tl.dot(q, k)

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        scores = tl.where((start_n + offs_n)[None, :] < N, scores, float("-inf"))

        # ── online softmax update ──
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores - m_new[:, None])

        # ── update running denominator + rescale accumulator BEFORE loading V ──
        # Doing the V load here (after the softmax computation) gives the
        # software pipeliner room to overlap the NEXT iteration's V load with
        # the CURRENT iteration's P @ V matmul. Empirically a few percent on
        # Ada/Blackwell, sometimes more.
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_bh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N, other=0.0)

        acc = tl.dot(p.to(v.dtype), v, acc)     # fused: acc += p @ v
        m_i = m_new

    # ── final normalization (deferred — FA-2) ──
    acc = acc / l_i[:, None]

    # ── store ──
    o_ptrs = O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N)
