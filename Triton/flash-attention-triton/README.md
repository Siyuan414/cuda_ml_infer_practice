# flash-attention-triton

A minimal, pedagogical [FlashAttention-2](https://arxiv.org/abs/2307.08691) forward kernel written in [Triton](https://github.com/triton-lang/triton). The goal is readability over peak throughput — if you want a production-grade kernel, use [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) or PyTorch's `scaled_dot_product_attention`.

## What's here

```
flash-attention-triton/
├── flash_attn_triton/
│   ├── kernel.py      # the @triton.jit FA-2 forward kernel
│   ├── wrapper.py     # Python entry point that launches the kernel
│   └── utils.py       # cuda-event-based benchmarking + diff helpers
├── benchmark/         # walltime and torch.profiler comparisons vs SDPA
└── tests/             # shape + numerical tests against F.scaled_dot_product_attention
```

## Install

Requires a CUDA GPU with compute capability 7.0+ (Volta or newer) and a CUDA-enabled PyTorch build.

```bash
pip install -e .
```

## Usage

```python
import torch
from flash_attn_triton import flash_attention

B, H, N, D = 2, 8, 1024, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attention(q, k, v)                 # non-causal
out = flash_attention(q, k, v, causal=True)    # causal
```

Shape contract: `(batch, num_heads, seq_len, head_dim)`, contiguous, fp16 or bf16, `head_dim ∈ {16, 32, 64, 128, 256}`.

## Run the tests

```bash
pytest tests/
```

Tests require a CUDA GPU. They compare the Triton kernel against `torch.nn.functional.scaled_dot_product_attention`.

## Benchmark

```bash
python -m benchmark.benchmark_attention
python -m benchmark.profile_attention
```

`benchmark_attention.py` reports median walltime vs PyTorch SDPA across a range of sequence lengths. `profile_attention.py` dumps a torch.profiler table so you can see per-op breakdowns.

See [RESULTS.md](RESULTS.md) for the latest measurements and analysis.

## Scope & limitations

- Forward pass only. No backward, so this is for inference or fine-tuning pipelines where the gradient comes from elsewhere.
- Optional causal masking, no attention bias, no dropout.
- Not exhaustively tuned. Block sizes are a single reasonable default (`BLOCK_M=128, BLOCK_N=64`); you'd normally wrap the kernel in `@triton.autotune` for production use.

## References

- Dao et al., "FlashAttention" (NeurIPS 2022)
- Dao, "FlashAttention-2" (2023)
- The official Triton tutorial `06-fused-attention.py`

## License

MIT
