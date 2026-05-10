"""QuantDebugger: per-layer activation diff between an FP16 reference and a
quantized model, the way you'd build it for production quantization triage.

Output: a table sorted by cosine similarity ascending, so the layers most
distorted by quantization float to the top.

    layer_name                                | cos_sim | mse      | max|diff|
    model.layers.21.mlp.down_proj             |  0.9842 | 1.2e-02  | 0.341
    model.layers.0.self_attn.q_proj           |  0.9871 | 4.4e-03  | 0.183
    ...

Layer matching: we walk the FP16 model's `named_modules()` for every
nn.Linear, then look up the same name path in the quant model. This works
for both gptqmodel and autoawq checkpoints because both preserve module
paths when swapping nn.Linear -> QuantLinear.

Usage (see apply_quant_debugger.py for a worked example):

    dbg = QuantDebugger(ref_model, quant_model)
    rows = dbg.compare(input_ids, attention_mask=mask)   # accumulates stats
    dbg.report()                                         # prints sorted table
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerStats:
    cos_sum: float = 0.0
    cos_count: int = 0
    sq_err_sum: float = 0.0
    elem_count: int = 0
    max_abs_diff: float = 0.0

    @property
    def cosine_similarity(self) -> float:
        return self.cos_sum / max(self.cos_count, 1)

    @property
    def mse(self) -> float:
        return self.sq_err_sum / max(self.elem_count, 1)


class QuantDebugger:
    """Compare per-layer linear outputs between two structurally identical models.

    Args:
        ref_model: FP16 (or BF16) reference. Forward pass must produce the
            same shape outputs at every linear as `quant_model`.
        quant_model: Quantized counterpart. Module paths must match `ref_model`.
        layer_filter: Optional callable (name, module) -> bool. Defaults to
            "every nn.Linear in ref_model except lm_head". Override to widen
            (include lm_head) or narrow (only attention).
    """

    def __init__(
        self,
        ref_model: nn.Module,
        quant_model: nn.Module,
        layer_filter=None,
    ):
        self.ref_model = ref_model
        self.quant_model = quant_model
        self.layer_filter = layer_filter or self._default_filter

        self._linear_names = self._discover_linears()
        self._stats: Dict[str, LayerStats] = defaultdict(LayerStats)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _default_filter(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        # lm_head produces (B, T, vocab) which dwarfs everything else;
        # exclude unless the user explicitly wants it
        if name.endswith("lm_head"):
            return False
        return True

    def _discover_linears(self) -> List[str]:
        names = []
        for name, mod in self.ref_model.named_modules():
            if self.layer_filter(name, mod):
                names.append(name)
        if not names:
            raise RuntimeError(
                "No linear layers matched layer_filter on ref_model. "
                "Did you pass the wrapper instead of the inner HF model?"
            )
        return names

    @staticmethod
    def _resolve_module(model: nn.Module, dotted_name: str) -> Optional[nn.Module]:
        mod = model
        for part in dotted_name.split("."):
            if not hasattr(mod, part):
                return None
            mod = getattr(mod, part)
        return mod

    # ------------------------------------------------------------------ stats
    def reset(self) -> None:
        self._stats.clear()

    @staticmethod
    def _accumulate(stats: LayerStats, ref: torch.Tensor, q: torch.Tensor) -> None:
        # Both shaped (..., D). Flatten leading dims, cosine over last dim.
        ref_f = ref.reshape(-1, ref.shape[-1]).float()
        q_f   = q.reshape(-1, q.shape[-1]).float()
        cos = F.cosine_similarity(ref_f, q_f, dim=-1)   # (N,)
        stats.cos_sum   += cos.sum().item()
        stats.cos_count += cos.numel()

        diff = ref_f - q_f
        stats.sq_err_sum += (diff * diff).sum().item()
        stats.elem_count += diff.numel()
        stats.max_abs_diff = max(stats.max_abs_diff, diff.abs().max().item())

    # ------------------------------------------------------------------ run
    @torch.no_grad()
    def compare(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, LayerStats]:
        """Run both models on `input_ids`, accumulate per-layer stats.

        Call multiple times with different inputs to average over a corpus —
        stats accumulate; call .reset() to start fresh.
        """
        # Phase 1: capture ref activations to CPU (fp16 to halve memory).
        ref_cache: Dict[str, torch.Tensor] = {}

        def make_ref_hook(name: str):
            def hook(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                ref_cache[name] = t.detach().to(torch.float16).cpu()
            return hook

        ref_handles = []
        for name in self._linear_names:
            mod = self._resolve_module(self.ref_model, name)
            if mod is None:
                continue
            ref_handles.append(mod.register_forward_hook(make_ref_hook(name)))

        try:
            self._forward(self.ref_model, input_ids, attention_mask)
        finally:
            for h in ref_handles:
                h.remove()

        # Phase 2: run quant model, compare in-hook against ref_cache, free as we go.
        def make_quant_hook(name: str):
            def hook(_module, _inp, out):
                if name not in ref_cache:
                    return
                t = out[0] if isinstance(out, tuple) else out
                ref_t = ref_cache.pop(name).to(t.device, dtype=torch.float32)
                self._accumulate(self._stats[name], ref_t, t.detach())
            return hook

        quant_handles = []
        for name in self._linear_names:
            mod = self._resolve_module(self.quant_model, name)
            if mod is None:
                continue
            quant_handles.append(mod.register_forward_hook(make_quant_hook(name)))

        try:
            self._forward(self.quant_model, input_ids, attention_mask)
        finally:
            for h in quant_handles:
                h.remove()

        return dict(self._stats)

    @staticmethod
    def _forward(
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> None:
        # Some quant wrappers (gptqmodel) put the actual transformer at .model.
        # We may have been handed either; HF models accept input_ids as kw.
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        model(**kwargs)

    # ------------------------------------------------------------------ report
    def rows(self) -> List[Tuple[str, float, float, float]]:
        out = []
        for name, s in self._stats.items():
            out.append((name, s.cosine_similarity, s.mse, s.max_abs_diff))
        out.sort(key=lambda r: r[1])  # cosine ascending = worst first
        return out

    def report(self, top_n: Optional[int] = None) -> None:
        rows = self.rows()
        if top_n:
            rows = rows[:top_n]
        print(f"{'layer_name':<55} {'cos_sim':>9} {'mse':>11} {'max|diff|':>11}")
        print("-" * 90)
        for name, cos, mse, mad in rows:
            print(f"{name:<55} {cos:>9.4f} {mse:>11.3e} {mad:>11.4f}")

    # ------------------------------------------------------------------ extras
    def category_summary(self) -> Dict[str, Tuple[float, int]]:
        """Group layers by attention vs FFN and by depth band, average cos_sim.

        Helps answer: "is degradation concentrated in attention or in FFN?
        Early layers or late layers?" — which is exactly what the afternoon
        task is supposed to investigate.
        """
        # Categorize by both subsystem (attn vs mlp) and depth.
        layer_idx_for: Dict[str, int] = {}
        max_idx = 0
        for name in self._stats:
            # parse "model.layers.<i>.<sub>.<linear>"
            parts = name.split(".")
            try:
                i = parts.index("layers")
                idx = int(parts[i + 1])
                layer_idx_for[name] = idx
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                layer_idx_for[name] = -1

        bands = {"early (0-33%)": (0, max_idx // 3),
                 "mid (33-66%)":   (max_idx // 3 + 1, 2 * max_idx // 3),
                 "late (66-100%)": (2 * max_idx // 3 + 1, max_idx)}

        groups: Dict[str, List[float]] = defaultdict(list)
        for name, s in self._stats.items():
            sub = "attn" if "self_attn" in name else ("mlp" if "mlp" in name else "other")
            idx = layer_idx_for.get(name, -1)
            band = next((b for b, (lo, hi) in bands.items() if lo <= idx <= hi), "other")
            groups[f"{sub} / {band}"].append(s.cosine_similarity)

        summary = {}
        for k, vals in groups.items():
            summary[k] = (sum(vals) / len(vals), len(vals))
        return summary
