# Quantization Analysis Report — TinyLlama-1.1B

**Author:** Siyuan Zhang
**Date:** 2026-05
**Hardware:** RTX 5070 Ti (Blackwell, sm_120) on WSL2 Ubuntu, CUDA 12.8, PyTorch 2.11+cu128
**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
**Calibration corpus:** 128 samples from C4 (`en/c4-train.00000-of-01024.json.gz`)
**Evaluation corpus:** WikiText-2 test split (chunked at 2048 tokens, max 80 chunks)

---

## Summary table

| Recipe | PPL | Δ vs FP16 | Size (MB) | Tokens/sec | Notes |
|---|---|---|---|---|---|
| FP16 baseline | 7.800 | — | 2200 | 134.6 | Reference |
| INT4 GPTQ (gptqmodel) | 8.346 | +7.00% | 789 | 60.4 | desc_act=True, gs=128 |
| INT4 AWQ (autoawq) | 8.176 | +4.83% | 734 | 61.1 | KL=0.030, top-1 80% |
| INT4 AWQ (ModelOpt) | 8.276 | +6.11% | — | 37.3¹ | TRT-LLM exportable |
| INT8 SmoothQuant (ModelOpt) | 8.763 | +12.34% | — | 33.9¹ | W8A8, default α=0.5 |
| Mixed: 15 worst FP16 + 4-bit gs=32 | 8.071 | +3.48% | 954 | 65.9 | Best-accuracy quantized |

¹ ModelOpt rows: tok/s is in-PyTorch fake-quant overhead, NOT deployment speed. Real INT4/INT8 throughput requires TRT-LLM compilation, out of scope for this study. The other rows (GPTQ/AWQ/Mixed) use real Triton inference kernels and represent honest deployment numbers.

## Top-5 worst-cosine layers per recipe

**INT4 GPTQ (gptqmodel)** — concentrated in mid-stack `mlp.down_proj`:
1. `model.layers.11.mlp.down_proj` (cos=0.9514)
2. `model.layers.10.mlp.down_proj` (cos=0.9552)
3. `model.layers.9.mlp.down_proj` (cos=0.9557)
4. `model.layers.12.mlp.down_proj` (cos=0.9561)
5. `model.layers.8.mlp.down_proj` (cos=0.9566)

**INT4 AWQ (autoawq)** — late-stack `mlp.up_proj` outliers + same `down_proj` cluster:
1. `model.layers.21.mlp.up_proj` (cos=0.9467)
2. `model.layers.7.mlp.up_proj` (cos=0.9471)
3. `model.layers.11.mlp.down_proj` (cos=0.9481)
4. `model.layers.17.self_attn.o_proj` (cos=0.9511)
5. `model.layers.12.mlp.down_proj` (cos=0.9513)

Across every recipe the most-distorted layers cluster in **mid- and late-stack MLP**. Attention `q_proj` / `k_proj` are essentially noise-free (cos > 0.997 across the board). MLP layers are the hard problem for low-bit quantization on Llama-class models — consistent with published literature on 7B+ models.

## Key findings

### 1. FP16 is the fastest deployment for this model size

Quantization halves throughput here (135 → 60 tok/s) because the model fits comfortably in 16 GB VRAM. INT4 weight-only kernels still dequantize to FP16 before matmul, so for tiny matmuls (TinyLlama's 2048-dim hidden state) the dequant overhead dominates the bandwidth savings. The "INT4 makes things fast" intuition is true at scale but inverts at 1B-class models on workstation GPUs.

### 2. AWQ outperforms GPTQ on this model — opposite of debugger prediction

The per-layer activation-diff debugger said GPTQ was better than AWQ (lower mean MSE, tighter cosine). End-to-end PPL says AWQ is better by ~2 percentage points. The debugger captured *local* error magnitude; what matters for end-to-end loss is how those errors *interact* through residuals and gating. AWQ's worse per-layer numbers came from a few `up_proj` outliers that don't propagate as catastrophically as the cosine numbers suggested.

**Lesson:** per-layer sensitivity metrics are useful for diagnosing *which* layers fail, but you can't rank *recipes* by them. End-to-end task metrics are the only honest measurement.

### 3. SmoothQuant fails on this model size (+12.34%)

SmoothQuant is the worst recipe in the table, despite being the most sophisticated. Two probable reasons:

- **TinyLlama-1.1B doesn't have the activation-outlier structure SmoothQuant exploits.** The technique migrates "difficulty" from activations to weights via a per-channel scale, on the assumption that a small fraction of activation channels have outsized magnitudes. That assumption holds at 7B+ but is much weaker at 1B.
- **W8A8 (both weights and activations quantized) compounds error** in ways W4A16 (weight-only) doesn't. Even with more bits per element, quantizing both sides of every matmul is harder than quantizing just one.

This finding doesn't generalize — SmoothQuant is competitive with FP16 on Llama-2-7B and bigger. The recipe failure here is specific to model size.

### 4. Three "AWQ" implementations, three different PPLs

```
INT4 AWQ (autoawq):  8.176  +4.83%
INT4 AWQ (ModelOpt): 8.276  +6.11%
INT4 GPTQ (gptqmodel): 8.346  +7.00%
```

Same paper, same model, same calibration corpus, same bit width — 1.3 percentage points of spread between AWQ implementations alone. The library matters: implementations differ on weight clipping, scale-search method, what gets skipped (lm_head, embeddings), and accumulator precision. **Picking an algorithm doesn't fully determine your accuracy.** You also pick a library, and that's a meaningful choice.

### 5. Mixed-precision: half the PPL gap, ~30% size cost

Keeping the worst-15 cosine-ranked linear modules in FP16 (with `group_size=32` for the rest) cut the PPL bump from +7.0% (pure GPTQ) to +3.48% — roughly halved. Disk size grew 700 → 954 MB (~36%). Per percentage-point of PPL recovery, mixed precision is the most cost-effective single intervention available.

The cumulative-drift caveat is real: even the layers we kept in FP16 still appear in the worst-cosine list because their *inputs* came from quantized upstream layers. So "keep N worst linears in FP16" gives diminishing returns past ~15. Block-level skipping (whole `layers.X.*` in FP16) would do better per FP16 byte, at the cost of coarser size impact.

## Production recommendation

**For this exact model (TinyLlama-1.1B): ship FP16.** It's faster, smaller relative to KV cache, and accuracy-perfect. The "quantization wins" calculation isn't compelling at 1B-class models on workstation hardware.

**If quantization is required** (the assumption behind the original task, and what you'd actually face at 7B+):

| Constraint | Recommendation |
|---|---|
| Best accuracy/size trade | **Mixed-precision GPTQ** (15 FP16 + INT4 gs=32). +3.48% PPL at +36% size cost vs pure 4-bit. |
| Smallest disk size | **AWQ via autoawq.** +4.83% PPL at 734 MB. Best PPL among pure 4-bit recipes. |
| TRT-LLM serving target | **AWQ via ModelOpt.** Slightly worse PPL than autoawq (+6.11%), but cleanly serializes to TRT-LLM checkpoint and engages NVIDIA's production toolchain. |
| Throughput-critical (1B model) | **FP16 again.** Quantization is a net loss on throughput at this size on Blackwell. |
| Throughput-critical (7B+) | Recipe choice flips — INT4 with continuous batching (vLLM/TRT-LLM) genuinely wins, but that's outside this study. |

**Do NOT recommend SmoothQuant for TinyLlama-class models.** The +12.34% PPL hit is unjustifiable when AWQ at 4 bits gets +4.83%. Worth re-evaluating at 7B+ where the recipe was designed to operate.

The honest one-line answer for an interview question is: **"It depends on the deployment context. There's no universal winner — the recipe needs to match the model size, target hardware, and serving system."**

## Caveats and future work

- **Throughput numbers are HF `generate` at batch=1.** Production-realistic numbers come from continuous-batched serving (vLLM, TRT-LLM). Relative ordering should hold; absolute numbers will be 2-5× higher with proper batching.
- **WikiText-2 perplexity is one signal.** Downstream-task evaluation (lm-evaluation-harness on HellaSwag, ARC, MMLU, WinoGrande) is the honest "is the model still useful" measurement — out of scope here, but the next thing I'd run.
- **ModelOpt throughput here is fake-quant overhead.** The real comparison would compile a TRT-LLM engine and benchmark that.
- **Single calibration corpus (C4 web text).** Calibration distribution should match deployment distribution. For a chat-tuned model deployed on instruction data, calibrating on a chat corpus would likely improve all recipes.
- **Single hyperparameter setting per recipe.** SmoothQuant especially is sensitive to its `alpha` parameter; tuning alpha was not attempted here. The default α=0.5 may not fit this model.
- **Block-level mixed precision** (whole `layers.X.*` in FP16 vs picking individual linears) is a likely improvement over the per-linear approach used here, given the cumulative-drift problem identified in the analysis. Worth a follow-up experiment.

## Reproducing this work

See [`README.md`](./README.md) for setup and run commands. The two venvs (gptq and modelopt) are required because the libraries pin incompatible transformers versions.
