# Mini Inference Server — Senior ML Inference Engineer Plan

**Goal:** Build a production-style LLM inference server ("mini-vLLM"), demonstrating
end-to-end ownership from kernel to serving layer.

**Timeline:** ~3–4 months | **Platform:** RTX 5070 Ti (WSL) for development; Jetson Orin
Nano as the edge-deployment story.

**Why this plan:** At 3 YOE + MS, the gap to senior isn't breadth — it's one deep,
coherent system whose every design decision you can defend. Continuous batching, paged
KV cache, and speculative decoding are what senior inference interviews probe; almost no
mid-level candidate has built them.

**Working mode (from Stage 2 onward):** coaching, not delivery. For each piece: design
and tradeoffs get explained, comprehension gets checked with questions, then *you* write
the code and it gets reviewed. Stage 1 was written for you, and the lesson learned was
that code you didn't write isn't code you can defend.

---

## Baseline before this project

- Quantization study: GPTQ, AWQ, SmoothQuant, mixed precision, QAT fundamentals
- CUDA kernels: tiled matmul, flash attention (Triton), Nsight profiling
- Jetson Orin Nano: TRT Edge-LLM (28.2 tok/s decode, Qwen3-4B) vs llama.cpp (18.1 tok/s)
- Single-request TRT runtime in `../llama_jetson` — the seed for this project

---

## Stage 1 — Single-request engine ✅ COMPLETE

→ [benchmarks/STAGE1.md](benchmarks/STAGE1.md)

| | Delivered |
|---|---|
| S1.1 | One engine, two TRT optimization profiles: prefill (seq 1–2048, past=0), decode (seq=1, past 0–2047) |
| S1.2 | `KVCache` class — ping-pong buffers, zero copies per step |
| S1.3 | True batched prefill — whole prompt in one `enqueueV3` |
| S1.4 | GPU sampling (temperature / top-k / top-p), 1.7% overhead |
| S1.5 | Dimensions from `config.json`; device-resident attention mask |
| S1.6 | Verified vs HF: cosine 1.0000, 5/5 top-1; benchmark sweep with warmup |

**Results:** 41k tok/s prefill (512 tokens), 235 tok/s decode, 2048 context.

**Two bugs worth remembering:**

1. **KV must be contiguous.** TRT reads `past_key_values` as `[1, H, past, D]` with head
   *h* at `h·past·D` — a stride that changes every step. A fixed-capacity buffer strided
   by `max_seq` misreads every head after head 0. Symptom: fluent, wrong output. Fix:
   ping-pong, since `present` is already the layout `past` needs next.
2. **The pre-tokenizer is a regex with ordered alternatives**, not a categorizer.
   `1969` → `196`+`9` (digits capped at 3), `(n` → one chunk (rule 2 precedes rule 4).
   Getting this wrong shifted logits by KL 0.52 while still producing fluent text.

---

## Stage 2 — Continuous batching ← the core of the project

**Split into 2A and 2B deliberately.** Paged KV requires a custom attention kernel,
because TRT's attention demands a contiguous `past` and paging scatters KV across blocks.
Building the scheduler and that kernel simultaneously means every bug is ambiguous, with
no reference to compare against. So:

- **2A** — fixed slots, TRT's stock attention → a working multi-request server and a
  correctness baseline. Teaches continuous batching.
- **2B** — swap in paged KV + custom attention as a pure memory optimization, validated
  against 2A. Teaches paged attention.

### S2.0 — Batched engine spike ✅ COMPLETE

Rebuilt the engine with a batch range (min 1 / opt 4 / max 32) on every input, and
verified ragged batched decode with `lens = [500, 30, 200, 1]`.

**Verification design.** `isfinite()` is a worthless gate — it passes even when masking
leaks. Three invariance tests instead:

| Test | Question | Result |
|---|---|---|
| A | slot 1 batched vs alone | 0.234 |
| B | same batch, padding poisoned +1e4 vs −1e4 | **0.000000** |
| C | slot 1 at past=500 (padded) vs past=30 (not) | 0.234 |

B is the correctness gate and it is *exact*: flipping 470 positions of masked memory
changed nothing, so padding provably never reaches the output. A and C being identical
showed batch size doesn't affect numerics at all — only reduction length does. 0.234 on a
signal of max 14.66 is 1.6%: fp16 rounding from a 501-term softmax vs a 31-term one.

**Lesson: "numerically different" is not "incorrect."** A and C measure batch-composition
reproducibility, a real property of every batched server (vLLM included), not correctness.

**What the spike established for the design:**

- Per-row `position_ids` are mandatory — a slot's new token is at *its own* next position,
  not at `past`; passing `past` misrotates RoPE.
- The mask has three regions per row: `[1 × len_i][0 × (past − len_i)][1 × seq]`.
  TRT appends new KV at index `past`, so the new columns are always last.
- **The shared `past` is the fundamental limitation.** One `past = max(lens)` for the whole
  batch means every slot carries the longest sequence's length, and `past` grows by 1 each
  step for every slot regardless of its real length. One 2000-token request forces all 8
  slots to carry 2000 tokens of KV. *This is the motivation for 2B, derived rather than
  asserted.*

### Stage 2A — Continuous batching, fixed slots

- **S2.1** `BatchKVCache` — N slots with independent lengths; builds padded KV, per-row
  mask and per-row positions each step. Open question: how a newly admitted request gets
  prefilled, given profile 1 only accepts `seq=1` and the batch's `past` is already large.
- **S2.2** `Request` + `Scheduler` — waiting queue, running set, admission control,
  eviction on EOS/max-tokens, and rebuilding the batch every step so requests join
  mid-flight (this is what "continuous" means, vs static batching).
- **S2.3** Batched sampling — `[B, vocab]` lm_head GEMM, per-row sampling with per-request
  temperature/top-k/top-p.
- **S2.4** HTTP front-end — streaming tokens, cancellation.
- **S2.5** Verify + benchmark — concurrent output matches sequential (within
  batch-composition drift), throughput vs batch size, TTFT/TPOT under load, slot
  occupancy → `STAGE2.md`.

### Stage 2B — Paged KV + custom attention

- **S2.6** Block allocator — fixed-size blocks (e.g. 16 tokens), free list, per-sequence
  block table.
- **S2.7** Split the ONNX graph so attention is ours: the engine keeps QKV projections and
  the MLP; a custom CUDA kernel does attention, reading KV through the block table.
- **S2.8** Paged attention kernel — the flash-attention work from `../Triton` is the
  starting point; the new part is block-table indirection.
- **S2.9** Verify against 2A (same outputs) and measure: memory per request, max
  concurrency at fixed VRAM, throughput vs 2A.

## Stage 3 — Speculative decoding (2–3 weeks)

- **S3.1** Draft model (1B drafts for 3B) + verification: accept/reject draft tokens
  against target logits.
- **S3.2** Measure acceptance rate vs draft length, and where it helps (long generation)
  vs hurts (short answers).

## Stage 4 — Benchmark + write-up (2 weeks)

- **S4.1** Head-to-head vs vLLM: TTFT, TPOT, goodput vs QPS, latency–throughput Pareto.
- **S4.2** One deep technical post on the scheduler and paged-cache design decisions.
- **S4.3** Repo polish: architecture diagram, benchmark tables, build docs.

---

## Open questions

- Prefill throughput peaks at 512 tokens (41k tok/s) and falls to 10k at 1900. Partly
  attention's O(n²), but the profile's `opt` is also 512 — rebuild with `opt=1024` to
  separate the two causes.
- `opt` batch is a placeholder at 4; set it from measured occupancy after S2.5.
- Max batch 32 at 2048 context needs ~4GB of KV with ping-pong. Fixed slots won't reach
  32; 2B is what makes high concurrency real.

## Explicitly deprioritized

- More quantization work (enough depth already)
- MLIR/TVM (different career track)
- k8s (learn on the job if a role requires it)
- Full Unicode pre-tokenizer (needs PCRE2/ICU; ASCII approximation verified 8/8)
