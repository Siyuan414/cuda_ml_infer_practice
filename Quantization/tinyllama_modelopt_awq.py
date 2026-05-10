"""Quantize TinyLlama-1.1B to 4-bit AWQ using NVIDIA ModelOpt, evaluate, and
export to both HuggingFace and TRT-LLM checkpoint formats.

This is the production NVIDIA path: ModelOpt is the same library used inside
TensorRT-LLM's official quantization pipeline. The point of running it here
isn't to beat your AutoAWQ result — it's to confirm that:
  (a) ModelOpt's INT4 AWQ produces accuracy similar to AutoAWQ (sanity)
  (b) The TRT-LLM export step works end-to-end (deployability)

Requires:
    pip install -U "nvidia-modelopt[hf,torch]"

Run inside the activated env:
    python tinyllama_modelopt_awq.py
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_OUT = "./tinyllama-1.1b-modelopt-awq-4bit-hf"
TRT_OUT = "./tinyllama-1.1b-modelopt-awq-4bit-trtllm"
DEVICE = "cuda"
N_CALIB = 128
CALIB_SEQ_LEN = 512
PPL_SEQ_LEN = 2048
PPL_MAX_CHUNKS = 80


# ----------------------------------------------------------------- calibration
def make_calib_loader(tokenizer):
    """A small list of tokenized C4 batches for ModelOpt's forward_loop."""
    ds = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True,
    )
    batches = []
    for row in ds:
        text = row["text"]
        if not (200 < len(text) < 2000):
            continue
        ids = tokenizer(
            text,
            return_tensors="pt",
            max_length=CALIB_SEQ_LEN,
            truncation=True,
        ).input_ids.to(DEVICE)
        batches.append(ids)
        if len(batches) >= N_CALIB:
            break
    return batches


# --------------------------------------------------------------------- ppl
@torch.no_grad()
def perplexity(model, tokenizer):
    """Standard chunked PPL on WikiText-2 test, same as eval_quant_vs_baseline."""
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(wt2["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    n_chunks = min(enc.size(1) // PPL_SEQ_LEN, PPL_MAX_CHUNKS)
    total_tokens, total_nll = 0, 0.0
    for i in range(n_chunks):
        ids = enc[:, i * PPL_SEQ_LEN : (i + 1) * PPL_SEQ_LEN]
        out = model(ids, labels=ids)
        n_tok = ids.size(1) - 1
        total_nll    += out.loss.float().item() * n_tok
        total_tokens += n_tok
    return math.exp(total_nll / total_tokens)


# --------------------------------------------------------------------- main
def main() -> None:
    print(f"Loading FP16 model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # ---- Optional: baseline FP16 PPL for the comparison row -----
    print("\nFP16 baseline perplexity...")
    t0 = time.time()
    ppl_fp16 = perplexity(model, tokenizer)
    print(f"  FP16 PPL = {ppl_fp16:.3f}  ({time.time()-t0:.1f}s)")

    # ---- Calibration data ---------------------------------------
    print(f"\nBuilding {N_CALIB}-sample calibration set from C4...")
    calib_batches = make_calib_loader(tokenizer)

    def forward_loop(m):
        # ModelOpt calls this during mtq.quantize() for activation stats.
        for ids in calib_batches:
            m(ids)

    # ---- Quantize -----------------------------------------------
    # mtq.INT4_AWQ_CFG is ModelOpt's predefined recipe matching the AWQ paper:
    # 4-bit weights, group_size=128, sym, with activation-aware per-channel
    # scaling. The same algorithm as autoawq, NVIDIA's reimplementation.
    cfg = mtq.INT4_AWQ_CFG
    print("\nApplying INT4 AWQ via ModelOpt...")
    print(f"  config algorithm: {cfg.get('algorithm', 'awq_lite')}")
    t0 = time.time()
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"  quantization complete  ({time.time()-t0:.1f}s)")

    # ---- Inline eval --------------------------------------------
    print("\nQuantized perplexity...")
    t0 = time.time()
    ppl_quant = perplexity(model, tokenizer)
    print(f"  ModelOpt INT4-AWQ PPL = {ppl_quant:.3f}  ({time.time()-t0:.1f}s)")
    delta_pct = 100.0 * (ppl_quant - ppl_fp16) / ppl_fp16

    # ---- Inline throughput (model still in memory; export-format-agnostic)
    from inline_benchmark import measure_and_dump
    measure_and_dump(
        model, tokenizer,
        label="modelopt-int4-awq",
        json_path="results/bench_modelopt-int4-awq.json",
        ppl=ppl_quant,
        delta_pct=delta_pct,
    )

    # ---- HF-format export (so eval/inference scripts can load it) -----
    print(f"\nExporting HF-format checkpoint -> {HF_OUT}")
    try:
        from modelopt.torch.export import export_hf_checkpoint
        export_hf_checkpoint(model, export_dir=HF_OUT)
        tokenizer.save_pretrained(HF_OUT)
        print("  HF export OK")
    except Exception as e:
        print(f"  HF export FAILED: {e}")

    # ---- TRT-LLM export (the actual production path) ------------
    print(f"\nExporting TRT-LLM checkpoint -> {TRT_OUT}")
    try:
        from modelopt.torch.export import export_tensorrt_llm_checkpoint
        export_tensorrt_llm_checkpoint(
            model,
            decoder_type="llama",          # TinyLlama uses Llama architecture
            dtype=torch.float16,
            export_dir=TRT_OUT,
            inference_tensor_parallel=1,
            inference_pipeline_parallel=1,
        )
        # Sanity-check: enumerate the export contents
        files = sorted(Path(TRT_OUT).rglob("*"))
        sizes = sum(f.stat().st_size for f in files if f.is_file())
        print(f"  TRT-LLM export OK: {len([f for f in files if f.is_file()])} files, "
              f"{sizes/1024**2:.1f} MB")
        for f in files[:8]:
            if f.is_file():
                print(f"    {f.relative_to(TRT_OUT)}  ({f.stat().st_size/1024:.0f} KB)")
    except Exception as e:
        print(f"  TRT-LLM export FAILED: {e}")

    # ---- Summary -------------------------------------------------
    print("\n" + "=" * 60)
    print(f"FP16 baseline PPL:           {ppl_fp16:.3f}")
    print(f"ModelOpt INT4-AWQ PPL:       {ppl_quant:.3f}  ({delta_pct:+.2f}%)")
    print()
    print("Compare against:")
    print("  pure 4-bit GPTQ (gptqmodel):    +7.00%")
    print("  pure 4-bit AWQ (gptqmodel):     ~+9% (predicted)")
    print("  mixed 15-FP16 + 4-bit gs=32:    +3.48%")


if __name__ == "__main__":
    main()
