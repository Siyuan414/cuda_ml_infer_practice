# llm_server — LLM Inference Server from Scratch

A production-style LLM serving engine in C++/CUDA/TensorRT, built up in stages:
single-request engine → paged KV cache + continuous batching → speculative decoding →
benchmarks vs vLLM.

See [INFERENCE_ENGINE_PLAN.md](INFERENCE_ENGINE_PLAN.md) for the full roadmap and task
breakdown.

## Status

- [x] **Stage 1** — Single-request engine → [benchmarks/STAGE1.md](benchmarks/STAGE1.md)
      41k tok/s prefill, 235 tok/s decode, verified against HF (cosine 1.0000, 5/5 top-1)
- [x] **Stage 2A** — Continuous batching → [benchmarks/STAGE2.md](benchmarks/STAGE2.md)
      947 tok/s peak (5.2× over single-request), 24/24 outputs identical to sequential
- [x] **Stage 2B** — Paged KV + custom decode path (no TensorRT)
      Block allocator (86% vs 11% KV utilization), paged attention kernel,
      hand-written forward pass verified layer-by-layer against HF (rel err ~0.005)
- [ ] Stage 2B.9 — Wire paged path into the scheduler; measure concurrency at fixed VRAM
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
src/   runtime.cpp          Stage 1: single-request TRT runtime
       batch_runtime.cpp    Stage 2A: continuous batching over TRT
       kv_cache.h           single-sequence KV (ping-pong)
       batch_kv_cache.h     N fixed slots + scatter
       scheduler.h          request lifecycle + admission policy (no CUDA)
       block_allocator.h    paged blocks, free list, block tables (no CUDA)
       decode_layer.h       Stage 2B: hand-written forward pass, no TensorRT
       weights.h            raw fp16 tensor loader
       tokenizer.h          byte-level BPE, LLaMA-3 pre-tokenizer rules
       model_config.h       dimensions from config.json
       argmax.cuh           CUB greedy argmax

kernels/ sampling.cuh       temperature / top-k / top-p on device
         batched_pick.cuh   one segmented argmax for B sequences
         kv_scatter.cuh     fixed-slot KV scatter (Stage 2A)
         layer_kernels.cuh  RMSNorm, RoPE, SiLU-mul, embedding, residual
         paged_attention.cuh  decode attention with block-table indirection

tools/  export_onnx.py, build_engine.py       build the TRT engine
        export_weights.py, dump_reference.py  build the custom decode path
        bench.py, bench_batch.py, bench_paged_attention.cu
        verify_logits.py, verify_tokenizer.py, verify_decode_path.cu
        test_block_allocator.cpp, test_paged_attention.cu, test_layer_kernels.cu
```

## Tests

```bash
g++  -std=c++17 -I src tools/test_block_allocator.cpp -o build/test_alloc
nvcc -std=c++17 -I kernels -I src tools/test_paged_attention.cu \
     -o build/test_paged --extended-lambda -arch=sm_120
nvcc -std=c++17 -I kernels tools/test_layer_kernels.cu -o build/test_layers -arch=sm_120
nvcc -std=c++17 -I src -I kernels tools/verify_decode_path.cu \
     -o build/verify_decode -lcublas --extended-lambda -arch=sm_120
```

## Origin

Starting point was the single-request TRT runtime in `../llama_jetson`
(C++ BPE tokenizer, GPU argmax, zero per-step malloc). That project remains the
edge-deployment reference (Jetson Orin Nano, TRT Edge-LLM vs llama.cpp).

Requires CUDA 12+, TensorRT 10.x (the pip `tensorrt` version must match the
system `libnvinfer` the binary links, or engine deserialization fails), cuBLAS.
