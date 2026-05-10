"""Throughput benchmark: tokens/sec for any HF-loadable checkpoint.

Two phases per run:
  - prefill: process the input prompt (compute-bound)
  - decode:  generate one token at a time (memory-bandwidth bound)

For LLM serving, decode tokens/sec is the metric that matters — it's what
shapes per-user latency. Prefill matters for time-to-first-token but is
usually a smaller fraction of total work for non-trivial generation lengths.

Why not vLLM? vLLM is the right answer for production-realistic numbers
(continuous batching, PagedAttention) but installing it is a 2GB compile
on Blackwell. HF generate is 3 lines of code, gives you tokens/sec on the
*same hardware path* you'd use for offline eval, and the ratios between
recipes (which is what you want for the report) are stable across both.

Run:
    python benchmark_throughput.py ./tinyllama-1.1b-gptq-4bit
    python benchmark_throughput.py TinyLlama/TinyLlama-1.1B-Chat-v1.0  # FP16

Use the --json flag to dump results to a file the aggregator can read.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "In recent years, large language models have shown remarkable performance "
    "across a wide range of natural language understanding tasks. The next "
    "frontier is making them efficient enough to deploy at scale, which "
    "requires careful attention to"
)
GEN_TOKENS = 128
N_RUNS = 5            # measurement runs (after warm-up)
N_WARMUP = 2
DEVICE = "cuda"


def load_model(path: str):
    """Load via gptqmodel for gptqmodel checkpoints, HF directly for FP16.

    ModelOpt-saved checkpoints are NOT supported here — they require modelopt
    installed at load time and use a non-standard "quantization_config: modelopt"
    in config.json that vanilla transformers doesn't understand. For those,
    use the inline throughput measurement in the modelopt scripts themselves
    (see inline_benchmark.measure_and_dump).
    """
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "quantize_config.json")
        hf_cfg_path = os.path.join(path, "config.json")

        # gptqmodel-style checkpoint: has its own quantize_config.json.
        if os.path.exists(cfg_path):
            from gptqmodel import BACKEND, GPTQModel
            meta = json.load(open(cfg_path))
            method = meta.get("quant_method", "gptq").lower()
            backend = BACKEND.GEMM_TRITON if method == "awq" else BACKEND.TRITON
            model = GPTQModel.load(path, device=DEVICE, backend=backend)
            return model, getattr(model, "tokenizer", None)

        # ModelOpt-style HF checkpoint: detect and bail out helpfully.
        if os.path.exists(hf_cfg_path):
            cfg = json.load(open(hf_cfg_path))
            qcfg = cfg.get("quantization_config", {}) or {}
            if qcfg.get("quant_method") == "modelopt":
                raise RuntimeError(
                    f"{path} is a ModelOpt-saved checkpoint and isn't loadable "
                    f"by vanilla transformers in this env. Use the inline "
                    f"throughput measurement built into the modelopt scripts "
                    f"(both tinyllama_modelopt_*.py now write a "
                    f"bench_modelopt-*.json automatically)."
                )

    # Fall through: treat as a standard HF model id or directory.
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    return model, None


@torch.no_grad()
def benchmark(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    n_input = ids.shape[1]

    # Warm-up
    for _ in range(N_WARMUP):
        _ = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    # Measurement runs
    decode_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        decode_times.append(elapsed)

    n_generated = out.shape[1] - n_input
    mean_t = sum(decode_times) / len(decode_times)
    median_t = sorted(decode_times)[len(decode_times) // 2]
    return {
        "prompt_tokens": n_input,
        "generated_tokens": n_generated,
        "mean_seconds": mean_t,
        "median_seconds": median_t,
        "tokens_per_sec_mean": n_generated / mean_t,
        "tokens_per_sec_median": n_generated / median_t,
        "all_runs_seconds": decode_times,
    }


def vram_used_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="HF model id or local path to quantized dir")
    ap.add_argument("--max-new-tokens", type=int, default=GEN_TOKENS)
    ap.add_argument("--json", help="Write results to this JSON file")
    ap.add_argument("--label", help="Label for this run (saved into JSON)")
    args = ap.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, _maybe_tok = load_model(args.checkpoint)
    # Tokenizer comes from the model dir for quant checkpoints, or the model id for FP16
    tok_src = args.checkpoint if os.path.isdir(args.checkpoint) else args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

    # gptqmodel wrappers don't carry a tokenizer; the AutoTokenizer call above is
    # what we use either way.

    print(f"Model loaded. VRAM after load: {vram_used_mb():.1f} MB")

    print(f"Benchmarking decode throughput "
          f"({N_WARMUP} warm-up + {N_RUNS} measurement runs, "
          f"{args.max_new_tokens} new tokens, prompt={len(PROMPT)} chars)...")
    result = benchmark(model, tokenizer, PROMPT, args.max_new_tokens)

    # Disk size if it's a directory
    disk_mb = None
    if os.path.isdir(args.checkpoint):
        disk_mb = sum(
            f.stat().st_size for f in Path(args.checkpoint).rglob("*") if f.is_file()
        ) / 1024**2

    print()
    print(f"=== {args.label or args.checkpoint} ===")
    print(f"  prompt tokens             : {result['prompt_tokens']}")
    print(f"  generated tokens          : {result['generated_tokens']}")
    print(f"  mean wall time / run      : {result['mean_seconds']:.3f}s")
    print(f"  median wall time / run    : {result['median_seconds']:.3f}s")
    print(f"  tokens/sec (mean)         : {result['tokens_per_sec_mean']:.1f}")
    print(f"  tokens/sec (median)       : {result['tokens_per_sec_median']:.1f}")
    if disk_mb is not None:
        print(f"  checkpoint disk size (MB) : {disk_mb:.1f}")
    print(f"  VRAM allocated (MB)       : {vram_used_mb():.1f}")

    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "label": args.label or args.checkpoint,
            "max_new_tokens": args.max_new_tokens,
            "disk_size_mb": disk_mb,
            "vram_allocated_mb": vram_used_mb(),
            **result,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
