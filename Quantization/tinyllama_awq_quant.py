"""Quantize TinyLlama-1.1B to 4-bit AWQ using the canonical `autoawq` library.

This uses the original MIT AWQ implementation (the `awq` / `autoawq` package),
not gptqmodel's integrated AWQ. The two are interoperable at the checkpoint
level (both produce the GEMM packing format), but the surface API and
quantization-time code paths are different.

Run inside the activated env:
    pip install autoawq         # one-time
    python tinyllama_awq_quant.py
"""
from __future__ import annotations

from awq import AutoAWQForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = "./tinyllama-1.1b-awq-4bit"
N_CALIB = 128

# autoawq's quant config is a plain dict, not a dataclass.
QUANT_CONFIG = {
    "zero_point": True,    # asymmetric quant; AWQ recommends this for 4-bit
    "q_group_size": 128,   # scale group (channel-wise within each group)
    "w_bit": 4,
    "version": "GEMM",     # GEMM = standard AWQ kernel layout (Marlin-compatible)
}


def build_calibration_set(n: int = N_CALIB) -> list[str]:
    """Return n cleaned text samples from C4 for activation statistics."""
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


def main() -> None:
    print(f"Loading FP16 model: {MODEL_ID}")
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    calib = build_calibration_set()
    print(f"loaded {len(calib)} calibration samples")

    print("Running AWQ scale search + quantization...")
    model.quantize(
        tokenizer,
        quant_config=QUANT_CONFIG,
        calib_data=calib,
        max_calib_samples=N_CALIB,
        max_calib_seq_len=512,
    )

    model.save_quantized(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"saved 4-bit AWQ checkpoint to {OUT_DIR}")

    # Smoke-test reload + generate.
    # autoawq's `from_quantized` selects its own kernel; on Blackwell without
    # CUDA 12.8 system toolkit you may need to pass `fuse_layers=False` and
    # rely on the GEMM kernel that ships prebuilt with autoawq-kernels.
    print("\nReload + smoke-test generation:")
    # autoawq's from_quantized only accepts "auto" / "balanced" / "balanced_low_0"
    # / "sequential" as device_map strings (it goes through accelerate). Use "auto".
    qmodel = AutoAWQForCausalLM.from_quantized(
        OUT_DIR,
        device_map="auto",
        fuse_layers=False,   # avoid extra JIT compilation on first load
    )
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = qmodel.generate(**ids, max_new_tokens=20, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
