"""Quantize TinyLlama-1.1B to INT8 using SmoothQuant via NVIDIA ModelOpt.

SmoothQuant (Xiao et al. 2022) is a different beast from GPTQ/AWQ. Instead of
just quantizing weights, it migrates "difficulty" from activations to weights
via a learned per-channel scale, then quantizes both to INT8. The result is
genuine W8A8 quantization, where activations also live in INT8 at runtime —
unlike weight-only INT4 GPTQ/AWQ, which dequantize weights to FP16 before
matmul.

Practical implications vs INT4 weight-only:
  - Less aggressive size reduction: INT8 is 2x compression vs FP16 (vs 4x for
    INT4). TinyLlama goes from 2.2GB FP16 -> ~1.1GB INT8 vs ~0.7GB INT4.
  - Better throughput on H100/A100/Blackwell that have native INT8 tensor
    cores: matmul itself runs in INT8 instead of FP16. INT4 weight-only still
    pays for FP16 matmul.
  - Generally less PPL bump than 4-bit; SmoothQuant W8A8 typically lands
    within 1-2% of FP16 for Llama-class models.

Run inside the modelopt_env:
    source modelopt_env/bin/activate
    python tinyllama_modelopt_smoothquant.py
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_OUT = "./tinyllama-1.1b-modelopt-smoothquant-int8-hf"
TRT_OUT = "./tinyllama-1.1b-modelopt-smoothquant-int8-trtllm"
DEVICE = "cuda"
N_CALIB = 128
CALIB_SEQ_LEN = 512
PPL_SEQ_LEN = 2048
PPL_MAX_CHUNKS = 80


def make_calib_loader(tokenizer):
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
            text, return_tensors="pt",
            max_length=CALIB_SEQ_LEN, truncation=True,
        ).input_ids.to(DEVICE)
        batches.append(ids)
        if len(batches) >= N_CALIB:
            break
    return batches


@torch.no_grad()
def perplexity(model, tokenizer):
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(wt2["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    n_chunks = min(enc.size(1) // PPL_SEQ_LEN, PPL_MAX_CHUNKS)
    total_tokens, total_nll = 0, 0.0
    for i in range(n_chunks):
        ids = enc[:, i * PPL_SEQ_LEN : (i + 1) * PPL_SEQ_LEN]
        out = model(ids, labels=ids)
        n_tok = ids.size(1) - 1
        total_nll += out.loss.float().item() * n_tok
        total_tokens += n_tok
    return math.exp(total_nll / total_tokens)


def main() -> None:
    print(f"Loading FP16 model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print("\nFP16 baseline perplexity...")
    t0 = time.time()
    ppl_fp16 = perplexity(model, tokenizer)
    print(f"  FP16 PPL = {ppl_fp16:.3f}  ({time.time()-t0:.1f}s)")

    print(f"\nBuilding {N_CALIB}-sample calibration set from C4...")
    calib_batches = make_calib_loader(tokenizer)

    def forward_loop(m):
        for ids in calib_batches:
            m(ids)

    # SmoothQuant config: W8A8, with the per-channel migration step run
    # automatically during calibration. The "smoothquant" part is the
    # algorithm field; the cfg also sets up INT8 quantizers for both
    # weights and activations.
    cfg = mtq.INT8_SMOOTHQUANT_CFG
    print("\nApplying INT8 SmoothQuant via ModelOpt...")
    print(f"  algorithm: {cfg.get('algorithm', 'smoothquant')}")
    t0 = time.time()
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"  quantization complete  ({time.time()-t0:.1f}s)")

    print("\nQuantized perplexity...")
    t0 = time.time()
    ppl_quant = perplexity(model, tokenizer)
    print(f"  ModelOpt INT8-SmoothQuant PPL = {ppl_quant:.3f}  ({time.time()-t0:.1f}s)")
    delta_pct = 100.0 * (ppl_quant - ppl_fp16) / ppl_fp16

    # ---- Inline throughput (model still in memory; export-format-agnostic)
    from inline_benchmark import measure_and_dump
    measure_and_dump(
        model, tokenizer,
        label="modelopt-int8-smoothquant",
        json_path="results/bench_modelopt-int8-smoothquant.json",
        ppl=ppl_quant,
        delta_pct=delta_pct,
    )

    print(f"\nExporting HF-format checkpoint -> {HF_OUT}")
    try:
        from modelopt.torch.export import export_hf_checkpoint
        export_hf_checkpoint(model, export_dir=HF_OUT)
        tokenizer.save_pretrained(HF_OUT)
        print("  HF export OK")
    except Exception as e:
        print(f"  HF export FAILED: {e}")

    print(f"\nExporting TRT-LLM checkpoint -> {TRT_OUT}")
    try:
        from modelopt.torch.export import export_tensorrt_llm_checkpoint
        export_tensorrt_llm_checkpoint(
            model,
            decoder_type="llama",
            dtype=torch.float16,
            export_dir=TRT_OUT,
            inference_tensor_parallel=1,
            inference_pipeline_parallel=1,
        )
        files = sorted(Path(TRT_OUT).rglob("*"))
        sz = sum(f.stat().st_size for f in files if f.is_file())
        print(f"  TRT-LLM export OK: {len([f for f in files if f.is_file()])} files, {sz/1024**2:.1f} MB")
    except Exception as e:
        print(f"  TRT-LLM export FAILED: {e}")

    print("\n" + "=" * 60)
    print(f"FP16 baseline PPL:                {ppl_fp16:.3f}")
    print(f"ModelOpt INT8-SmoothQuant PPL:    {ppl_quant:.3f}  ({delta_pct:+.2f}%)")
    print()
    print("Stack ranking so far:")
    print("  pure 4-bit GPTQ (gptqmodel):    +7.00%")
    print("  mixed 15-FP16 + 4-bit gs=32:    +3.48%")
    print(f"  INT8 SmoothQuant (this run):    {delta_pct:+.2f}%")


if __name__ == "__main__":
    main()
