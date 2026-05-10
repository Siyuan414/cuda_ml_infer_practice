"""Aggregate per-recipe results into one comparison table.

Reads the JSONs the eval / benchmark / debugger scripts produce, joins them
by recipe name, and prints a final table you can paste into the report.

Expected files (paths configurable via --results-dir):
    bench_<recipe>.json                  - from benchmark_throughput.py
    quant_debugger_layers_<recipe>.json  - from apply_quant_debugger.py
    ppl_<recipe>.txt                     - one-line file with the PPL number
                                            (or set ppl_overrides below)

Run:
    python aggregate_results.py
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecipeResult:
    name: str
    method: str
    ppl: Optional[float] = None
    delta_pct: Optional[float] = None
    size_mb: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    worst_layers: List[str] = field(default_factory=list)
    notes: str = ""


# Hardcode known results so the user can paste partial data and still see the table.
# Add/edit these as you collect numbers.
KNOWN_RESULTS = [
    RecipeResult(
        name="fp16-baseline",
        method="FP16 (no quant)",
        ppl=7.800, delta_pct=0.0,
        size_mb=2200.0,
        notes="Reference",
    ),
    RecipeResult(
        name="gptq-4bit-gs128",
        method="INT4 GPTQ (gptqmodel)",
        ppl=8.346, delta_pct=7.00,
    ),
    RecipeResult(
        name="awq-4bit-autoawq",
        method="INT4 AWQ (autoawq)",
        ppl=8.176, delta_pct=4.83,
        notes="KL=0.0295, top-1 80%/5",
    ),
    RecipeResult(
        name="modelopt-int4-awq",
        method="INT4 AWQ (ModelOpt)",
        notes="tok/s is fake-quant overhead, not deployment speed",
    ),
    RecipeResult(
        name="modelopt-int8-smoothquant",
        method="INT8 SmoothQuant (ModelOpt)",
        notes="tok/s is fake-quant overhead, not deployment speed",
    ),
    RecipeResult(
        name="gptq-mixed-15fp16",
        method="Mixed: 15 worst FP16, rest INT4 GPTQ gs=32",
        ppl=8.071, delta_pct=3.48,
    ),
]


def load_bench(results_dir: str, recipe: str) -> Optional[dict]:
    p = os.path.join(results_dir, f"bench_{recipe}.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


def load_worst_layers(results_dir: str, recipe: str, top_n: int = 5) -> List[str]:
    """Read the debugger JSON and return the N worst-cosine layer names."""
    candidates = [
        os.path.join(results_dir, f"quant_debugger_layers_{recipe}.json"),
        # naming variant we've already produced
        os.path.join(results_dir, f"quant_debugger_layers_{recipe.replace('4bit','4bit')}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            rows = json.load(open(p))
            rows.sort(key=lambda r: r["cosine_similarity"])
            return [r["layer_name"] for r in rows[:top_n]]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    # Try to enrich each known recipe with file-loaded data
    for r in KNOWN_RESULTS:
        bench = load_bench(args.results_dir, r.name)
        if bench:
            r.tokens_per_sec = bench.get("tokens_per_sec_median")
            if r.size_mb is None:
                r.size_mb = bench.get("disk_size_mb")
            # Pull PPL/delta from bench JSON if not already set in KNOWN_RESULTS
            if r.ppl is None and bench.get("ppl") is not None:
                r.ppl = bench["ppl"]
            if r.delta_pct is None and bench.get("delta_pct") is not None:
                r.delta_pct = bench["delta_pct"]
        worst = load_worst_layers(args.results_dir, r.name)
        if worst:
            r.worst_layers = worst

    # Special case: known existing debugger files
    name_to_recipe = {r.name: r for r in KNOWN_RESULTS}
    if not name_to_recipe["gptq-4bit-gs128"].worst_layers:
        name_to_recipe["gptq-4bit-gs128"].worst_layers = load_worst_layers(
            args.results_dir, "gptq4bit"
        )
    if not name_to_recipe["awq-4bit-autoawq"].worst_layers:
        name_to_recipe["awq-4bit-autoawq"].worst_layers = load_worst_layers(
            args.results_dir, "awp4bit"
        )

    # ---- Print main comparison table ----
    print("=" * 110)
    print(f"{'recipe':<35} {'PPL':>8} {'Δ%':>7} {'size MB':>9} {'tok/s':>9} {'note':<35}")
    print("-" * 110)
    for r in KNOWN_RESULTS:
        ppl = f"{r.ppl:.3f}" if r.ppl is not None else "—"
        d   = f"{r.delta_pct:+.2f}" if r.delta_pct is not None else "—"
        sz  = f"{r.size_mb:.0f}" if r.size_mb is not None else "—"
        tps = f"{r.tokens_per_sec:.1f}" if r.tokens_per_sec is not None else "—"
        print(f"{r.method:<35} {ppl:>8} {d:>7} {sz:>9} {tps:>9} {r.notes:<35}")
    print("=" * 110)

    # ---- Per-recipe top-5 worst layers ----
    print("\nTop-5 worst-cosine layers per recipe:\n")
    for r in KNOWN_RESULTS:
        if not r.worst_layers:
            continue
        print(f"  {r.method}:")
        for i, n in enumerate(r.worst_layers, 1):
            print(f"    {i}. {n}")
        print()


if __name__ == "__main__":
    main()
