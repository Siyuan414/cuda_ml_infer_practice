"""Mixed-precision GPTQ: keep the worst-cosine layers in FP16, quantize the
rest to 4-bit GPTQ.

This is the standard production-quantization workflow. Instead of quantizing
everything uniformly, you spend FP16 budget on the few layers that suffer
most under 4-bit, and aggressively quantize the rest. For TinyLlama-1.1B,
this typically claws back most of the +7% perplexity bump in exchange for
a ~10-20% increase in checkpoint size.

The picks come from quant_debugger_layers_gptq4bit.json, which the debugger
writes after running. Each row gives a (layer_name, cosine_similarity, ...)
tuple; we pick the bottom-N by cosine and exclude them from quantization.

Run inside the activated env:
    # produce a checkpoint that keeps the worst 15 linears in FP16
    python mixed_precision_gptq.py --keep-fp16 15

    # evaluate it
    python eval_quant_vs_baseline.py ./tinyllama-1.1b-gptq-mixed-15fp16
"""
from __future__ import annotations

import argparse
import json
import os
import re

from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEBUG_JSON = "quant_debugger_layers_gptq4bit.json"
N_CALIB = 128


def build_calibration_set(n: int = N_CALIB) -> list[str]:
    """Same calibration corpus as the original GPTQ run, for a fair comparison."""
    ds = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True,
    )
    samples = []
    for row in ds:
        text = row["text"]
        if 200 < len(text) < 2000:
            samples.append(text)
            if len(samples) >= n:
                break
    return samples


def pick_worst_layers(json_path: str, n_keep_fp16: int) -> list[str]:
    """Return the n_keep_fp16 worst-cosine layer names."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"{json_path} not found. Run apply_quant_debugger.py against your "
            "GPTQ checkpoint first to generate per-layer stats."
        )
    rows = json.load(open(json_path))
    rows.sort(key=lambda r: r["cosine_similarity"])  # worst first
    return [r["layer_name"] for r in rows[:n_keep_fp16]]


def build_dynamic_skip_config(layer_names: list[str]) -> dict:
    """Build gptqmodel's `dynamic` dict with exact-match exclusion patterns.

    Pattern format:
        "-:<regex>"  -> skip quantization for matching modules (kept FP16)
        "+:<regex>"  -> apply per-layer overrides (different bits/group_size)

    PCRE matching is anchored at the start by default, so we add `$` at the
    end to force exact-match (otherwise "model.layers.1" would also match
    "model.layers.10", "model.layers.11", ...).
    """
    dynamic = {}
    for name in layer_names:
        pattern = "-:" + re.escape(name) + r"$"
        dynamic[pattern] = {}
    return dynamic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-fp16",
        type=int,
        default=15,
        help="How many worst-cosine linear modules to keep in FP16 (default: 15, ~10%%).",
    )
    ap.add_argument(
        "--debug-json",
        default=DEBUG_JSON,
        help="Per-layer debugger output to use for layer selection.",
    )
    args = ap.parse_args()

    fp16_layers = pick_worst_layers(args.debug_json, args.keep_fp16)
    print(f"Keeping {len(fp16_layers)} layers in FP16 (worst cosine first):")
    for i, n in enumerate(fp16_layers, 1):
        print(f"  {i:>2}. {n}")

    out_dir = f"./tinyllama-1.1b-gptq-mixed-{args.keep_fp16}fp16"

    quant_cfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=True,
        damp_percent=0.01,
        sym=True,
        dynamic=build_dynamic_skip_config(fp16_layers),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    calib = build_calibration_set()
    print(f"\nLoaded {len(calib)} calibration samples")

    print("Loading model...")
    model = GPTQModel.load(MODEL_ID, quant_cfg)
    print("Quantizing (FP16 layers will be skipped)...")
    model.quantize(calib, batch_size=2)
    model.save(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nSaved mixed-precision GPTQ checkpoint to {out_dir}")

    # Quick size report
    sizes = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            sizes.append(os.path.getsize(os.path.join(root, f)))
    total_mb = sum(sizes) / (1024 ** 2)
    print(f"Checkpoint size on disk: {total_mb:.1f} MB")
    print(f"  (compare: pure FP16 ~2200 MB, pure 4-bit GPTQ ~700 MB)")

    print("\nNext step: evaluate vs FP16 baseline")
    print(f"    python eval_quant_vs_baseline.py {out_dir}")
    print("\nCompare PPL against your original GPTQ run (baseline=7.800, gptq=8.346).")


if __name__ == "__main__":
    main()
