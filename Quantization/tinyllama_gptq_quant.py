"""Quantize TinyLlama-1.1B to 4-bit GPTQ using gptqmodel (the maintained
fork of auto-gptq with Blackwell / RTX 50-series support).

Run inside the activated env:
    python tinyllama_gptq_quant.py
"""
from __future__ import annotations

from datasets import load_dataset
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = "./tinyllama-1.1b-gptq-4bit"
N_CALIB = 128  # 128-512 is typical for small models


def build_calibration_set(tokenizer, n: int = N_CALIB):
    """A small, diverse text corpus to estimate per-channel activation stats."""
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
    # gptqmodel accepts list[str] OR list[dict] with input_ids/attention_mask.
    # Strings are simplest and let gptqmodel handle truncation.
    return samples


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    calib = build_calibration_set(tokenizer)
    print(f"loaded {len(calib)} calibration samples")

    quant_cfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=True,        # better accuracy, slightly slower quantization
        damp_percent=0.01,    # Hessian damping
        sym=True,
    )

    model = GPTQModel.load(MODEL_ID, quant_cfg)
    model.quantize(calib, batch_size=2)
    model.save(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved 4-bit GPTQ checkpoint to {OUT_DIR}")

    # Reload + smoke-test generation.
    # Use Triton backend until CUDA 12.8 toolkit is installed system-wide
    # (Marlin JIT-compiles CUDA and needs nvcc >= 12.8 for sm_120).
    model = GPTQModel.load(OUT_DIR, backend=BACKEND.TRITON)
    prompt = "The capital of France is"
    out = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        max_new_tokens=20,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
