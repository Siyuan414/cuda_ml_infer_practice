# Mini Inference Server — Senior ML Inference Engineer Plan

**Goal:** Build a production-style LLM inference server ("mini-vLLM") on top of the existing
`llama_jetson` C++ runtime, demonstrating end-to-end ownership from kernel to serving layer.

**Project home:** `cuda_ml_infer_practice/llm_server/` — seeded with `runtime.cpp`,
`tokenizer.h`, `argmax.cuh` copied from `llama_jetson/src/`. Line references below are
to `src/runtime.cpp` in this folder.

**Timeline:** ~3–4 months | **Platform:** RTX 5070 Ti (WSL) for development; Jetson Orin Nano as edge deployment story

**Why this plan:** At 3 YOE + MS, the gap to senior isn't breadth — it's one deep, coherent
system you built and can defend every design decision of. Continuous batching, paged KV cache,
and speculative decoding are what senior inference interviews probe; almost no mid-level
candidate has built them.

---

## Current baseline (already done)

- C++ TRT runtime: BPE tokenizer, GPU argmax, zero per-step malloc — 303 tok/s decode, 3.30 ms/tok, 910 GB/s (94.8% of peak) on 5070 Ti
- Jetson Orin Nano: TRT Edge-LLM (28.2 tok/s decode, Qwen3-4B) vs llama.cpp (18.1 tok/s)
- Quantization study: GPTQ, AWQ, SmoothQuant, mixed precision, QAT fundamentals
- CUDA kernels: tiled matmul, flash attention (Triton), profiling

## Known limitations of current runtime (`src/runtime.cpp`)

| Issue | Location | Fix |
|---|---|---|
| Prefill loops token-by-token (runs at decode speed) | line 310 | S1.1 + S1.3 |
| MAX_PAST = 63 hardcoded context | line 47 | S1.1 |
| Greedy argmax only | `lm_head_argmax` | S1.4 |
| Model constants baked in (constexpr) | lines 42–47 | S1.5 |
| Host-side mask std::vector rebuilt every step | line 191 | S1.5 |
| Single request only, no scheduler | everywhere | Stage 2 |

---

## Stage 1 — Single-request engine, done right (3–4 weeks)

Generalize the existing runtime into a clean single-request engine.

- **S1.1** Rebuild ONNX/TRT engine with dynamic seq_len and two optimization profiles:
  prefill (1–2048) and decode (seq_len=1). Removes MAX_PAST=63.
- **S1.2** `KVCache` class — owns per-layer K/V device buffers (max_seq_len=2048), tracks
  length, `append()` / `reset()`. **Design the interface so paged backing storage
  (Stage 2) slots in without touching the engine loop.** This is the critical seam.
- **S1.3** True prefill: whole prompt in one `enqueueV3` using the prefill profile.
  Expect 10–50× prefill speedup.
- **S1.4** GPU sampling: temperature, top-k, top-p as CUDA kernels; only the sampled
  token id returns to host.
- **S1.5** Load model config from HF `config.json` (works for any LLaMA-family model);
  keep attention mask on device, extend with a tiny kernel.
- **S1.6** Verify: exact-match vs HF transformers greedy output (3 prompts); report
  prefill tok/s, decode tok/s, TTFT, p50/p95 vs the 303 tok/s baseline → `STAGE1.md`.

**Order:** S1.1 → S1.3 → S1.2 → S1.4/S1.5 → S1.6

## Stage 2 — Continuous batching (4–6 weeks) ← the core senior skill

- **S2.1** Paged KV cache: fixed-size blocks (e.g. 16 tokens), block allocator with
  free list, block table per sequence (vLLM PagedAttention design)
- **S2.2** Request scheduler: request queue, admission (fits in free blocks?),
  preemption/eviction policy, iteration-level scheduling (new requests join the
  batch every decode step)
- **S2.3** Batched decode over ragged sequences: variable kv lengths per batch entry.
  Requires either a TRT engine with batch dim + per-sequence mask, or a custom
  attention path — decide after S2.1 profiling.
- **S2.4** Multi-request front-end: simple HTTP or gRPC endpoint, streaming responses,
  request cancellation
- **S2.5** Verify: correctness under concurrency (outputs identical to sequential runs),
  throughput vs batch size curve, memory utilization vs naive per-request allocation

## Stage 3 — Speculative decoding (2–3 weeks)

- **S3.1** Draft model (LLaMA-3.2-1B drafts for 3B, or a tiny draft head) + verification
  step: accept/reject sampled draft tokens against target model logits
- **S3.2** Measure: acceptance rate vs draft length, end-to-end speedup on 5070 Ti,
  where it helps (long generation) vs hurts (short answers)

## Stage 4 — Benchmark + write-up (2 weeks)

- **S4.1** Head-to-head vs vLLM: TTFT, TPOT, goodput at varying QPS; latency-throughput
  Pareto curves
- **S4.2** One deep technical blog post on the scheduler/paged-cache design decisions
- **S4.3** Clean up repo: README with architecture diagram, benchmark tables, build docs

---

## Explicitly deprioritized

- More quantization work (enough depth already)
- MLIR/TVM (different career track — compiler engineer)
- k8s (learn on the job if a role requires it)
- Building agents (day-job prototyping covers exposure)

## Session task list (Stage 1)

Tasks S1.1–S1.6 are tracked in the Cowork task list; recreate from this file if starting
a fresh session.
