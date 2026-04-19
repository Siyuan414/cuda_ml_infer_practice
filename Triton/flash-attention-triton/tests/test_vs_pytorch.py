import pytest
import torch
import torch.nn.functional as F

from flash_attn_triton.wrapper import flash_attention

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# (B, H, N, D) shapes chosen to cover:
#   - B>1, H>1  (exercises the batch/head grid axis)
#   - N not divisible by BLOCK_M  (exercises tail masking)
#   - different head dims
SHAPES = [
    (1, 1, 256, 64),
    (2, 4, 512, 64),
    (2, 8, 1024, 128),
    (1, 4, 500, 64),     # N not a multiple of BLOCK_M=128
]


@cuda_only
@pytest.mark.parametrize("B,H,N,D", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(B, H, N, D, causal):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = flash_attention(q, k, v, causal=causal)

    max_abs = (out.float() - ref.float()).abs().max().item()
    # fp16 tolerance — SDPA and Triton do reductions in different orders.
    assert max_abs < 2e-2, f"max |Δ|={max_abs:.3e} exceeds 2e-2 (B={B} H={H} N={N} D={D} causal={causal})"
