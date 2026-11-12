"""Quantization parity (ADR-0039): does an exported ONNX variant keep the torch model's quality?

Inputs are 4-way verbalizer logits (rows x 4) for the same rows from the torch model and from an exported
variant, the gold labels, and the calibration (temperature, threshold) fitted on val for the torch model.
Everything is pure numpy; no model is loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.eval.metrics import Pred, asr, auroc, fpr
from breachlab.train.scoring import probs_from_logits

Arr = NDArray[np.float64]

# Parity rules used to pick the winning variant (ADR-0039): the AUROC may drop at most 0.003 and the FPR at the
# tuned threshold may rise at most 0.5 percentage points (absolute) versus torch on the same rows.
MAX_AUROC_DROP = 0.003
MAX_FPR_RISE = 0.005
MAX_ASR_RISE = 0.01  # a uniform logit shift keeps AUROC but moves attacks under the threshold


@dataclass(frozen=True)
class Parity:
    n: int
    n_benign: int
    n_attack: int
    auroc_torch: float
    auroc_variant: float
    fpr_torch: float
    fpr_variant: float
    asr_torch: float
    asr_variant: float
    label_agreement: float
    max_abs_logit_diff: float
    ok: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def p_unsafe(logits: Arr, temperature: float) -> Arr:
    return np.asarray(1.0 - probs_from_logits(logits, temperature)[:, 0], dtype=np.float64)


def _preds(y_true: Sequence[int], p: Arr, threshold: float) -> list[Pred]:
    return [
        Pred(y_true=int(y), y_pred=int(pi >= threshold), p_unsafe=float(pi))
        for y, pi in zip(y_true, p, strict=True)
    ]


def variant_parity(
    y_true: Sequence[int],
    torch_logits: Arr,
    variant_logits: Arr,
    temperature: float,
    threshold: float,
) -> Parity:
    """Compare a variant with torch on the same rows at the torch model's calibration."""
    tl = np.asarray(torch_logits, dtype=np.float64)
    vl = np.asarray(variant_logits, dtype=np.float64)
    if tl.shape != vl.shape or tl.shape[0] != len(y_true):
        raise ValueError(
            f"shape mismatch: torch {tl.shape}, variant {vl.shape}, labels {len(y_true)}"
        )
    pt, pv = p_unsafe(tl, temperature), p_unsafe(vl, temperature)
    a, b = _preds(y_true, pt, threshold), _preds(y_true, pv, threshold)
    agree = float(np.mean((pt >= threshold) == (pv >= threshold)))
    res = dict(
        auroc_torch=auroc(a),
        auroc_variant=auroc(b),
        fpr_torch=fpr(a),
        fpr_variant=fpr(b),
    )
    asr_t, asr_v = asr(a), asr(b)
    ok = (
        res["auroc_torch"] - res["auroc_variant"] <= MAX_AUROC_DROP
        and res["fpr_variant"] - res["fpr_torch"] <= MAX_FPR_RISE
        and asr_v - asr_t <= MAX_ASR_RISE
    )
    n_att = int(sum(y_true))
    return Parity(
        n=len(y_true),
        n_benign=len(y_true) - n_att,
        n_attack=n_att,
        asr_torch=asr_t,
        asr_variant=asr_v,
        label_agreement=agree,
        max_abs_logit_diff=float(np.max(np.abs(tl - vl))),
        ok=bool(ok),
        **res,
    )


def pick_winner(
    parities: dict[str, Parity], p95_ms: dict[str, float], p50_ms: dict[str, float]
) -> str | None:
    """Fastest (by p50, ties by p95) variant whose parity is ok; None when every variant fails parity."""
    ok = [v for v, p in parities.items() if p.ok and v in p50_ms]
    if not ok:
        return None
    return min(ok, key=lambda v: (p50_ms[v], p95_ms.get(v, float("inf"))))


def _file_summary(files_mb: Mapping[str, float]) -> dict[str, Any]:
    """The exporter writes one external-data file per tensor; keep the total and the biggest few."""
    top = sorted(files_mb.items(), key=lambda kv: -kv[1])[:3]
    return {"total_mb": sum(files_mb.values()), "largest": dict(top)}


def summarize_export3(
    export3: Mapping[str, Any],
    variant_logits: Callable[[str, str], Arr | None],
    torch_logits: Callable[[str], Arr],
    calibration: Callable[[str], Mapping[str, float]],
    y_true: Sequence[int],
) -> dict[str, Any]:
    """Per-model parity + CPU latency summary of a multi-model export kernel (scripts/kernel_export_body.py).

    ``export3`` maps each `which` to ``{variant: {tok512_c1: {...}, tok128_c1: {...}}}`` (or an ``export_error``);
    ``variant_logits(which, variant)`` returns the first-N-gold-row logits or None. Variants whose export or
    scoring failed are listed with their error and never win.
    """
    out: dict[str, Any] = {}
    for spec in export3.get("specs", []):
        which = str(spec[0] if isinstance(spec, list | tuple) else spec)
        entry = export3.get(which, {})
        res: dict[str, Any] = {
            "base_model": entry.get("base_model"),
            "cpu_threads": export3.get("cpu_threads"),
            "requests": export3.get("requests"),
            "export_error": entry.get("export_error"),
            "files_mb": _file_summary(entry.get("files_mb") or {}),
            "variants": {},
        }
        out[which] = res
        if entry.get("export_error"):
            continue
        cal = dict(calibration(which))
        res["calibration"] = cal
        t_all = torch_logits(which)
        parities: dict[str, Parity] = {}
        for variant in ("fp32", "dyn_int8"):
            v = entry.get(variant, {})
            lg = variant_logits(which, variant)
            cell: dict[str, Any] = {k: v[k] for k in ("tok512_c1", "tok128_c1") if k in v}
            if "error" in v:
                cell["error"] = v["error"]
            if lg is not None:
                n = lg.shape[0]
                parities[variant] = variant_parity(
                    list(y_true[:n]), t_all[:n], lg, cal["temperature"], cal["threshold"]
                )
                cell["parity"] = parities[variant].as_dict()
            res["variants"][variant] = cell
        cells = res["variants"]
        p50 = {k: c["tok512_c1"]["p50_ms"] for k, c in cells.items() if "tok512_c1" in c}
        p95 = {k: c["tok512_c1"]["p95_ms"] for k, c in cells.items() if "tok512_c1" in c}
        res["winner"] = pick_winner(parities, p95, p50)
    return out
