# llm_server — LLM Inference Server from Scratch

A production-style LLM serving engine in C++/CUDA/TensorRT, built up in stages:
single-request engine → paged KV cache + continuous batching → speculative decoding →
benchmarks vs vLLM.

See [INFERENCE_ENGINE_PLAN.md](INFERENCE_ENGINE_PLAN.md) for the full roadmap and task
breakdown.

## Status

- [x] **Stage 1** — Single-request engine → [benchmarks/STAGE1.md](benchmarks/STAGE1.md)
      41k tok/s prefill, 235 tok/s decode, verified against HF (cosine 1.0000, 5/5 top-1)
- [ ] Stage 2 — Continuous batching (paged KV cache, scheduler, HTTP front-end)
- [ ] Stage 3 — Speculative decoding
- [ ] Stage 4 — Benchmarks vs vLLM + write-up

## Quick start

```bash
python tools/export_onnx.py --model <hf_model_dir>   # → onnx/
python tools/build_engine.py                         # → engine/
cmake -B build && cmake --build build -j

./build/runtime \
  --engine engine/llama1b_fp16.trt \
  --lm-head onnx/lm_head_weight.bin \
  --tokenizer onnx/tokenizer.json \
  --prompt "The key insight about transformers is" \
  --max-new-tokens 64 [--temperature 0.8 --top-k 50 --top-p 0.95]
```

Verify and benchmark:

```bash
python tools/verify_tokenizer.py                     # 8/8 vs HF tokenizers
python tools/verify_logits.py --model <hf_model_dir> # 5/5 top-1, cosine 1.0000
python tools/bench.py                                # perf sweep
```

## Layout

```
src/       runtime.cpp      engine, decode loop, CLI
           kv_cache.h       KV memory + TRT binding (Stage 2 seam)
           tokenizer.h      byte-level BPE, LLaMA-3 pre-tokenizer rules
           model_config.h   dimensions from config.json
           argmax.cuh       CUB greedy argmax
kernels/   sampling.cuh     temperature / top-k / top-p, fully on device
tools/     export_onnx.py, build_engine.py, bench.py, verify_*.py
benchmarks/ STAGE1.md       curated report (bench.py writes perf_raw.md)
```

## Origin

Starting point was the single-request TRT runtime in `../llama_jetson`
(C++ BPE tokenizer, GPU argmax, zero per-step malloc). That project remains the
edge-deployment reference (Jetson Orin Nano, TRT Edge-LLM vs llama.cpp).

Requires CUDA 12+, TensorRT 10.x (the pip `tensorrt` version must match the
system `libnvinfer` the binary links, or engine deserialization fails), cuBLAS.
