import pytest
import torch

from flash_attn_triton.wrapper import flash_attention

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda_only
def test_shape_preserved():
    q = torch.randn(1, 1, 128, 64, device="cuda", dtype=torch.float16)
    out = flash_attention(q, q, q)
    assert out.shape == q.shape


@cuda_only
def test_output_is_finite_and_nonzero():
    """Guards against a silently-broken kernel that returns all zeros or NaN."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = flash_attention(q, k, v)
    assert torch.isfinite(out).all(), "output contains NaN or inf"
    assert out.abs().max() > 0, "output is all zero"
