"""torch.profiler breakdown of the Triton FlashAttention-2 forward kernel."""

# See benchmark_attention.py for why this is here.
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from flash_attn_triton.wrapper import flash_attention


def profile_run():
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # warmup: excludes Triton's JIT compile time from the profile
    for _ in range(5):
        flash_attention(q, k, v)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            with record_function("triton_flash_attention"):
                flash_attention(q, k, v)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    profile_run()
