"""flash_attn_triton — a minimal, pedagogical FlashAttention-2 forward kernel in Triton."""

from .kernel import flash_attn_fwd
from .wrapper import flash_attention

__all__ = ["flash_attention", "flash_attn_fwd"]
