"""Apply QuantDebugger to FP16 TinyLlama vs the 4-bit GPTQ checkpoint.

Answers (when run):
  - Which layers degrade most under quantization?
  - Are the worst layers in attention or MLP?
  - Are they early in the stack or late?

Run inside the activated env:
    python apply_quant_debugger.py                                   # GPTQ default
    python apply_quant_debugger.py ./tinyllama-1.1b-awq-4bit         # any checkpoint
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_debugger import QuantDebugger

BASELINE_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_QUANT_DIR = "./tinyllama-1.1b-gptq-mixed-15fp16"   # produced by tinyllama_gptq_quant.py
DEVICE = "cuda"
N_BATCHES = 8        # how many sample batches to average over
SEQ_LEN = 256        # truncate prompts to this many tokens


def pick_backend(quant_dir: str):
    cfg = os.path.join(quant_dir, "quantize_config.json")
    method = "gptq"
    if os.path.exists(cfg):
        method = json.load(open(cfg)).get("quant_method", "gptq").lower()
    return BACKEND.GEMM_TRITON if method == "awq" else BACKEND.TRITON


def calibration_batches(tokenizer, n_batches: int, seq_len: int):
    """Stream a few WikiText-2 windows; each batch is one (1, seq_len) tensor."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate, then chunk
    full = "\n\n".join(ds["text"])
    ids = tokenizer(full, return_tensors="pt").input_ids[0]
    batches = []
    for i in range(n_batches):
        chunk = ids[i * seq_len : (i + 1) * seq_len]
        if chunk.numel() < seq_len:
            break
        batches.append(chunk.unsqueeze(0).to(DEVICE))
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("quant_dir", nargs="?", default=DEFAULT_QUANT_DIR)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASELINE_ID, use_fast=True)

    print(f"Loading FP16 baseline:  {BASELINE_ID}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASELINE_ID, torch_dtype=torch.float16
    ).to(DEVICE).eval()

    backend = pick_backend(args.quant_dir)
    print(f"Loading quant model:    {args.quant_dir}  (backend={backend.value})")
    quant_wrapper = GPTQModel.load(args.quant_dir, device=DEVICE, backend=backend)
    # gptqmodel's wrapper has the actual HF model at .model; that's what
    # has the named modules we want to hook. ref_model already IS the HF model.
    quant_inner = getattr(quant_wrapper, "model", quant_wrapper)
    quant_inner.eval()

    print("Building debugger...")
    dbg = QuantDebugger(ref_model=ref_model, quant_model=quant_inner)

    print(f"Comparing across {N_BATCHES} batches of seq_len={SEQ_LEN}...")
    batches = calibration_batches(tokenizer, N_BATCHES, SEQ_LEN)
    for i, b in enumerate(batches):
        dbg.compare(b)
        print(f"  batch {i + 1}/{len(batches)} done")

    # -------- per-layer table, worst first --------
    print("\n=== Per-layer activation diff, worst first ===")
    dbg.report(top_n=20)

    # -------- attention vs MLP, early vs late --------
    print("\n=== Subsystem x depth, mean cosine similarity ===")
    summary = dbg.category_summary()
    for k, (cos, n) in sorted(summary.items(), key=lambda kv: kv[1][0]):
        print(f"  {k:<25} cos={cos:.4f}   ({n} layers)")

    # -------- save full table for the afternoon mixed-precision step --------
    out_path = "quant_debugger_layers.json"
    rows = [{"layer_name": n, "cosine_similarity": c, "mse": m, "max_abs_diff": d}
            for n, c, m, d in dbg.rows()]
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote per-layer table -> {out_path}")
    print("(Use this to pick the bottom-N layers to keep in FP16 for the")
    print(" mixed-precision experiment.)")


if __name__ == "__main__":
    main()
