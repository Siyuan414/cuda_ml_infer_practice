# LLM Quantization Comparison — TinyLlama-1.1B

End-to-end comparison of four post-training quantization recipes on TinyLlama-1.1B,
plus a per-layer activation-diff debugger for triaging quantization-induced error.

**Final report:** [REPORT.md](./REPORT.md)

## What's in here

| File | Purpose |
|---|---|
| `quant_debugger.py` | `QuantDebugger` class — per-layer activation diff (cosine / MSE / max\|diff\|) between FP16 and a quantized model. |
| `apply_quant_debugger.py` | Driver that runs `QuantDebugger` against a checkpoint and writes `quant_debugger_layers_*.json`. |
| `tinyllama_gptq_quant.py` | INT4 GPTQ via gptqmodel, group_size=128. |
| `tinyllama_awq_quant.py` | INT4 AWQ via autoawq. |
| `tinyllama_modelopt_awq.py` | INT4 AWQ via NVIDIA ModelOpt + TRT-LLM export. |
| `tinyllama_modelopt_smoothquant.py` | INT8 SmoothQuant (W8A8) via ModelOpt. |
| `mixed_precision_gptq.py` | Selective quantization — keeps worst-cosine linears in FP16. |
| `eval_quant_vs_baseline.py` | PPL / KL / top-1 agreement vs FP16 baseline. |
| `benchmark_throughput.py` | Tokens-per-second measurement (HF `generate`, batch=1). |
| `inline_benchmark.py` | Helper used by ModelOpt scripts to measure throughput before export. |
| `aggregate_results.py` | Joins per-recipe JSONs into the final comparison table. |
| `verify_gpu.py` | Sanity-check torch sees the GPU and gptqmodel imports cleanly. |
| `setup_gptq_env.sh` | Creates `gptq_env/` venv with PyTorch cu128, gptqmodel, autoawq, transformers, etc. |
| `setup_modelopt_env.sh` | Creates separate `modelopt_env/` venv (separate because library version pins are incompatible). |

## Hardware / software

- WSL2 Ubuntu 22.04, Windows host with NVIDIA driver ≥ 570.x
- RTX 50-series GPU (Blackwell, sm_120). Tested on 5070 Ti.
- CUDA Toolkit 12.8 (system-installed, for any kernel JIT compilation)
- Python 3.11, PyTorch 2.11+cu128

## Why two venvs

`nvidia-modelopt` pins `transformers < 5` while `gptqmodel >= 7` requires `transformers >= 5.4`. Forcing one venv to satisfy both leads to constant import breakage. Two venvs is the clean answer:

- `gptq_env/` — for gptqmodel, autoawq, debugger, eval, benchmark
- `modelopt_env/` — for ModelOpt scripts only

Switch between them via `source <env>/bin/activate`.

## Running the full study

```bash
# 1. one-time env setup
./setup_gptq_env.sh
./setup_modelopt_env.sh

# 2. quantize (in gptq_env)
source gptq_env/bin/activate
python tinyllama_gptq_quant.py
python tinyllama_awq_quant.py

# 3. quantize (in modelopt_env)
deactivate && source modelopt_env/bin/activate
python tinyllama_modelopt_awq.py             # also writes bench_modelopt-int4-awq.json
python tinyllama_modelopt_smoothquant.py     # also writes bench_modelopt-int8-smoothquant.json

# 4. accuracy eval (back in gptq_env)
deactivate && source gptq_env/bin/activate
python eval_quant_vs_baseline.py ./tinyllama-1.1b-gptq-4bit
python eval_quant_vs_baseline.py ./tinyllama-1.1b-awq-4bit

# 5. per-layer debugger (in gptq_env)
python apply_quant_debugger.py ./tinyllama-1.1b-gptq-4bit
python apply_quant_debugger.py ./tinyllama-1.1b-awq-4bit

# 6. mixed-precision (driven by the debugger output)
python mixed_precision_gptq.py --keep-fp16 15
python eval_quant_vs_baseline.py ./tinyllama-1.1b-gptq-mixed-15fp16

# 7. throughput on every recipe with a real inference kernel
python benchmark_throughput.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --label fp16-baseline --json bench_fp16-baseline.json
python benchmark_throughput.py ./tinyllama-1.1b-gptq-4bit          --label gptq-4bit-gs128 --json bench_gptq-4bit-gs128.json
python benchmark_throughput.py ./tinyllama-1.1b-awq-4bit           --label awq-4bit-autoawq --json bench_awq-4bit-autoawq.json
python benchmark_throughput.py ./tinyllama-1.1b-gptq-mixed-15fp16  --label gptq-mixed-15fp16 --json bench_gptq-mixed-15fp16.json

# 8. final comparison table
python aggregate_results.py
```

## Reading the results

After `aggregate_results.py` you should see a table with PPL / size / tok/s / worst-5-layers per recipe. The same data is written into `REPORT.md`. See that file for the analysis and production recommendation.

## Notes / known issues

- **ModelOpt's HF export is not loadable by vanilla transformers.** It uses a `quantization_config: modelopt` config that requires modelopt installed at load time. Throughput for those rows is therefore measured *inline* during quantization (in `inline_benchmark.measure_and_dump`), not via `benchmark_throughput.py`. The inline numbers reflect PyTorch fake-quant overhead, not real INT4/INT8 deployment speed — for production numbers you'd compile a TRT-LLM engine.
- **CUDA 12.8 toolkit is required** for Marlin kernel JIT compilation on Blackwell. Without it, the eval/benchmark scripts fall back to `BACKEND.TRITON` (slower but works).
- **gptqmodel and autoawq cannot share a transformers version with modelopt.** Don't `pip install` modelopt into `gptq_env`. The setup scripts enforce this separation.

## License

MIT.
