"""Compare a quantized model (GPTQ or AWQ) against its FP16 baseline.

Three measurements:

  1. Perplexity on WikiText-2 test (the standard LLM-quant benchmark).
  2. Logit-level agreement on a fixed set of prompts:
        - mean KL(baseline || quant) over the next-token distribution
        - top-1 / top-5 agreement of next-token predictions
  3. Side-by-side greedy generations on a few prompts (qualitative).

Run inside the activated env:
    python eval_quant_vs_baseline.py                          # default: GPTQ dir
    python eval_quant_vs_baseline.py ./tinyllama-1.1b-awq-4bit
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import GPTQModel, BACKEND

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASELINE_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_QUANT_DIR = "./tinyllama-1.1b-gptq-mixed-15fp16"   # produced by tinyllama_gptq_quant.py

DEVICE        = "cuda"
# Marlin is fastest but JIT-compiles CUDA at load and needs a system nvcc that
# knows about sm_120. If your /usr/bin/nvcc is < 12.8 it'll fail with
# "Unsupported gpu architecture 'compute_120'". Until CUDA 12.8 toolkit is
# installed we use Triton (no nvcc needed). The Triton kernel name is
# different for GPTQ vs AWQ, so we pick at runtime from the checkpoint's
# quantize_config.json. Set USE_MARLIN=True after upgrading CUDA toolkit.
USE_MARLIN = False
SEQ_LEN       = 2048          # context window used for perplexity
PPL_MAX_CHUNKS = 80           # cap eval chunks (full WT2 is ~100 chunks @2k ctx)
LOGIT_PROMPTS = [
    "The capital of France is",
    "In thermodynamics, the second law states that",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "She opened the door and saw",
    "The mitochondrion is the powerhouse of the",
]
GEN_PROMPTS = LOGIT_PROMPTS[:3]
GEN_TOKENS = 40


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@contextmanager
def timed(label: str):
    t0 = time.time()
    yield
    print(f"  [{label}] {time.time() - t0:.1f}s")


def load_baseline():
    print(f"Loading FP16 baseline: {BASELINE_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_ID, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    return model


def pick_backend(quant_dir: str) -> BACKEND:
    """Pick the right kernel backend for this checkpoint.

    GPTQ and AWQ use different Triton kernel names because they pack weights
    differently. Read quantize_config.json and branch on quant_method.
    """
    cfg_path = os.path.join(quant_dir, "quantize_config.json")
    method = "gptq"
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            method = json.load(f).get("quant_method", "gptq").lower()

    if USE_MARLIN:
        return BACKEND.MARLIN  # resolved to GPTQ_MARLIN or AWQ_MARLIN automatically
    return BACKEND.GEMM_TRITON if method == "awq" else BACKEND.TRITON


def load_quant(quant_dir: str):
    backend = pick_backend(quant_dir)
    print(f"Loading quant model:   {quant_dir}  (backend={backend.value})")
    model = GPTQModel.load(quant_dir, device=DEVICE, backend=backend)
    # gptqmodel returns a wrapped HF model; the underlying nn.Module is .model
    inner = getattr(model, "model", model)
    inner.eval()
    return model, inner


@torch.no_grad()
def perplexity(model_callable, tokenizer, dataset_text: str) -> float:
    """Sliding-chunk perplexity over a long string (WikiText-2 style)."""
    enc = tokenizer(dataset_text, return_tensors="pt").input_ids.to(DEVICE)
    total_tokens, total_nll = 0, 0.0
    n_chunks = min(enc.size(1) // SEQ_LEN, PPL_MAX_CHUNKS)
    for i in range(n_chunks):
        ids = enc[:, i * SEQ_LEN : (i + 1) * SEQ_LEN]
        # next-token loss = -log p(x_{t+1} | x_<=t), averaged over the chunk
        out = model_callable(ids, labels=ids)
        # `out.loss` is mean over (seq_len - 1) shifted positions
        n_tok = ids.size(1) - 1
        total_nll    += out.loss.float().item() * n_tok
        total_tokens += n_tok
    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def logit_agreement(baseline, quant, tokenizer, prompts) -> dict:
    """Compare next-token logits on the *last* position of each prompt."""
    kl_sum, top1_match, top5_match, n = 0.0, 0, 0, 0
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.to(DEVICE)
        b_logits = baseline(ids).logits[0, -1].float()
        q_logits = quant(ids).logits[0, -1].float()

        b_logp = F.log_softmax(b_logits, dim=-1)
        q_logp = F.log_softmax(q_logits, dim=-1)
        b_p    = b_logp.exp()

        # KL(baseline || quant)  =  sum_x b_p * (log b_p - log q_p)
        kl = (b_p * (b_logp - q_logp)).sum().item()
        kl_sum += kl

        b_top1 = b_logits.argmax().item()
        q_top1 = q_logits.argmax().item()
        if b_top1 == q_top1:
            top1_match += 1

        b_top5 = set(b_logits.topk(5).indices.tolist())
        q_top5 = set(q_logits.topk(5).indices.tolist())
        if b_top1 in q_top5:
            top5_match += 1
        n += 1
    return {
        "mean_kl": kl_sum / n,
        "top1_acc": top1_match / n,
        "top1_in_top5": top5_match / n,
        "n_prompts": n,
    }


@torch.no_grad()
def greedy_compare(baseline, quant, tokenizer, prompts):
    print("\n--- Greedy generation, baseline vs quant ---")
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").to(DEVICE)
        b = baseline.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False)
        q = quant.generate(**ids, max_new_tokens=GEN_TOKENS, do_sample=False)
        print(f"\nprompt: {p!r}")
        print("  base :", tokenizer.decode(b[0], skip_special_tokens=True))
        print("  quant:", tokenizer.decode(q[0], skip_special_tokens=True))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "quant_dir",
        nargs="?",
        default=DEFAULT_QUANT_DIR,
        help="Path to the quantized checkpoint (GPTQ or AWQ).",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASELINE_ID, use_fast=True)

    print("Loading WikiText-2 test split...")
    wt2 = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(wt2["text"])

    # ---- baseline ---------------------------------------------------------
    baseline = load_baseline()
    with timed("baseline ppl"):
        ppl_base = perplexity(baseline, tokenizer, text)
    print(f"FP16 baseline PPL: {ppl_base:.3f}")

    # We'll need both models loaded for logit comparison, so leave baseline on GPU
    # (TinyLlama 1.1B fp16 ~2.2GB; 4-bit GPTQ ~0.7GB -> fits easily on 16GB).

    # ---- quantized --------------------------------------------------------
    quant_wrapper, quant_inner = load_quant(args.quant_dir)
    # gptqmodel's wrapper exposes .generate; for forward pass on logits use inner
    with timed("quant ppl"):
        ppl_quant = perplexity(quant_inner, tokenizer, text)
    print(f"Quant 4bit PPL:    {ppl_quant:.3f}")

    delta = ppl_quant - ppl_base
    pct = 100.0 * delta / ppl_base
    print(f"\nDelta PPL: {delta:+.3f}  ({pct:+.2f}% vs FP16)")

    # ---- logit agreement --------------------------------------------------
    print("\nComputing logit agreement on fixed prompts...")
    agree = logit_agreement(baseline, quant_inner, tokenizer, LOGIT_PROMPTS)
    print(f"  mean KL(base || quant): {agree['mean_kl']:.4f} nats")
    print(f"  top-1 match            : {agree['top1_acc'] * 100:.1f}%")
    print(f"  base top-1 in quant top-5: {agree['top1_in_top5'] * 100:.1f}%")

    # ---- qualitative ------------------------------------------------------
    greedy_compare(baseline, quant_wrapper, tokenizer, GEN_PROMPTS)

    # ---- summary ----------------------------------------------------------
    print("\n=== Summary ===")
    print(f"baseline PPL : {ppl_base:8.3f}")
    print(f"quant    PPL : {ppl_quant:8.3f}  ({pct:+.2f}%)")
    print(f"top-1 agree  : {agree['top1_acc'] * 100:5.1f}% over {agree['n_prompts']} prompts")
    print(f"mean KL      : {agree['mean_kl']:.4f}")


if __name__ == "__main__":
    main()
