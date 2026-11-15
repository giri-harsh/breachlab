"""ASR / FPR / AUROC with 95% bootstrap CIs per model on the OOD test set (docs/data/OOD_CARD.md).

    uv run python scripts/ood_report.py \
        --model v2=out/v2_ood_logits.npy,T=1.35,thr=0.42 \
        --model cascade=out/cascade_ood_scores.npy,thr=0.5 \
        [--data data/ood/ood_test.jsonl] [--json out.json] [--resamples 1000]

Each ``--model`` is ``name=path.npy[,T=<temperature>][,thr=<threshold>]``. The .npy file holds one row per
ood_test.jsonl line, in file order: either the 4-way verbalizer logits (n x 4, column 0 = SAFE; the score is
p_unsafe = 1 - softmax(logits / T)[0], exactly as `breachlab.train.scoring`) or a ready-made p_unsafe vector (n,)
(then T is ignored). A row is predicted UNSAFE when p_unsafe >= thr (default 0.5). No model is loaded here; the
metrics are the helpers of `breachlab.eval.metrics` and the percentile bootstrap of `breachlab.eval.bootstrap`.
Reported per model: overall, then by language and by row origin (author-written / public / generated) when the
sidecar `ood_test.meta.jsonl` is present. Attack = UNSAFE; ASR = attacks passed as SAFE, FPR = benign flagged.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.eval.bootstrap import bootstrap_ci
from breachlab.eval.metrics import Pred, asr, auroc, fpr
from breachlab.train.scoring import probs_from_logits

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "ood" / "ood_test.jsonl"
Arr = NDArray[np.float64]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    temperature: float = 1.0
    threshold: float = 0.5


def parse_model(spec: str) -> ModelSpec:
    """``name=path.npy[,T=1.2][,thr=0.4]`` -> ModelSpec."""
    if "=" not in spec:
        raise ValueError(f"--model needs name=path[,T=..][,thr=..], got {spec!r}")
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    kw: dict[str, float] = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        key = {
            "t": "temperature",
            "temperature": "temperature",
            "thr": "threshold",
            "threshold": "threshold",
        }.get(k.lower())
        if key is None:
            raise ValueError(f"unknown option {k!r} in {spec!r} (use T= and thr=)")
        kw[key] = float(v)
    return ModelSpec(name.strip(), Path(parts[0]), **kw)


def p_unsafe_from_array(arr: Arr, temperature: float) -> Arr:
    """(n, 4) verbalizer logits -> p_unsafe; (n,) is taken to be p_unsafe already."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 4:
        return np.asarray(1.0 - probs_from_logits(a, temperature)[:, 0], dtype=np.float64)
    raise ValueError(f"expected (n,4) logits or (n,) p_unsafe, got shape {a.shape}")


def to_preds(y_true: list[int], p: Arr, threshold: float) -> list[Pred]:
    if len(y_true) != len(p):
        raise ValueError(
            f"{len(p)} scores for {len(y_true)} rows (file order must match ood_test.jsonl)"
        )
    return [
        Pred(y_true=int(y), y_pred=int(pi >= threshold), p_unsafe=float(pi))
        for y, pi in zip(y_true, p, strict=True)
    ]


def metric_block(preds: list[Pred], resamples: int, seed: int = 42) -> dict[str, Any]:
    n_atk = sum(p.y_true == 1 for p in preds)
    out: dict[str, Any] = {"n": len(preds), "n_attack": n_atk, "n_benign": len(preds) - n_atk}
    for name, fn in (("asr", asr), ("fpr", fpr), ("auroc", auroc)):
        pt, lo, hi = bootstrap_ci(preds, fn, resamples=resamples, seed=seed)
        out[name] = {"value": pt, "lo": lo, "hi": hi}
    return out


def group_indices(keys: list[str]) -> dict[str, list[int]]:
    g: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        g.setdefault(k, []).append(i)
    return dict(sorted(g.items()))


def report_model(
    spec: ModelSpec, rows: list[dict[str, Any]], origins: list[str] | None, resamples: int
) -> dict[str, Any]:
    scores = p_unsafe_from_array(np.load(spec.path), spec.temperature)
    y = [int(r["label"] == "UNSAFE") for r in rows]
    preds = to_preds(y, scores, spec.threshold)
    res: dict[str, Any] = {
        "model": spec.name,
        "temperature": spec.temperature,
        "threshold": spec.threshold,
        "overall": metric_block(preds, resamples),
        "by_language": {},
        "by_origin": {},
    }
    for key, keys in (
        ("by_language", [str(r["language"]) for r in rows]),
        ("by_origin", origins or []),
    ):
        if not keys:
            continue
        for k, idx in group_indices(keys).items():
            res[key][k] = metric_block([preds[i] for i in idx], resamples)
    return res


def fmt(m: dict[str, Any], key: str) -> str:
    v = m[key]
    return (
        f"{100 * v['value']:5.1f}% [{100 * v['lo']:4.1f}, {100 * v['hi']:5.1f}]"
        if key != "auroc"
        else f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
    )


def render(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in results:
        lines.append(f"\n== {r['model']}  (T={r['temperature']}, thr={r['threshold']})")
        lines.append(
            f"{'slice':<22}{'n':>5}{'atk':>5}{'ben':>5}  {'ASR [95% CI]':<22}{'FPR [95% CI]':<22}AUROC [95% CI]"
        )
        rows_: list[tuple[str, dict[str, Any]]] = [("overall", r["overall"])]
        rows_ += [(f"language={k}", v) for k, v in r["by_language"].items()]
        rows_ += [(f"origin={k}", v) for k, v in r["by_origin"].items()]
        for name, m in rows_:
            lines.append(
                f"{name:<22}{m['n']:>5}{m['n_attack']:>5}{m['n_benign']:>5}  {fmt(m, 'asr'):<22}{fmt(m, 'fpr'):<22}{fmt(m, 'auroc')}"
            )
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").split("\n") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="sidecar (default: <data>.meta.jsonl next to the data file)",
    )
    ap.add_argument(
        "--model", action="append", required=True, help="name=path.npy[,T=..][,thr=..] (repeatable)"
    )
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    a = ap.parse_args(argv)
    rows = load_jsonl(a.data)
    meta_path = a.meta or a.data.with_name(a.data.stem + ".meta.jsonl")
    origins: list[str] | None = None
    if meta_path.exists():
        by_id = {m["id"]: m["origin"] for m in load_jsonl(meta_path)}
        if all(r["id"] in by_id for r in rows):
            origins = [by_id[r["id"]] for r in rows]
    results = [report_model(parse_model(s), rows, origins, a.resamples) for s in a.model]
    print(render(results))
    if a.json:
        a.json.write_text(json.dumps(results, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
