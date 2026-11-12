"""Eval report (PRD 9.2-9.6): comparison table with CIs, McNemar, slices, worst misses, size-vs-quality
table, and the public ``scorecard.json``. Pure functions over aligned prediction lists."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from breachlab.data.schema import Row
from breachlab.eval.bootstrap import bootstrap_ci
from breachlab.eval.mcnemar import mcnemar
from breachlab.eval.metrics import (
    Pred,
    asr,
    confusion,
    fpr,
    prf,
    slice_report,
    summarize,
    worst_slice,
)


def _f1(p: list[Pred]) -> float:
    return prf(p)[2]


METRICS = {"asr": asr, "fpr": fpr, "f1": _f1}


@dataclass
class ModelEval:
    name: str
    preds: list[Pred]  # aligned with the gold rows
    p95_ms: float | None = None
    cost_per_1k_usd: float | None = None
    train_gpu_h: float | None = None
    note: str = ""
    seeds: list[dict[str, float]] = field(
        default_factory=list
    )  # per-seed summaries for mean +- std


def ci_cell(preds: list[Pred], key: str, resamples: int, seed: int) -> str:
    pt, lo, hi = bootstrap_ci(preds, METRICS[key], resamples=resamples, seed=seed)
    return f"{pt * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"


def correct(preds: list[Pred]) -> list[bool]:
    return [p.y_true == p.y_pred for p in preds]


def worst_misses(rows: list[Row], preds: list[Pred], n: int = 20) -> list[tuple[Row, Pred]]:
    """Attacks passed with the highest confidence of being safe, then false positives with highest p_unsafe."""
    fn = [(r, p) for r, p in zip(rows, preds, strict=True) if p.y_true == 1 and p.y_pred == 0]
    fp = [(r, p) for r, p in zip(rows, preds, strict=True) if p.y_true == 0 and p.y_pred == 1]
    fn.sort(key=lambda rp: rp[1].p_unsafe)
    fp.sort(key=lambda rp: -rp[1].p_unsafe)
    return (fn + fp)[:n]


def size_vs_quality(models: dict[str, ModelEval], names: list[str]) -> list[str]:
    lines = [
        "| model | F1 | ASR | FPR | p95 ms (CPU) | cost/1k | train GPU-h |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in names:
        m = models.get(n)
        if m is None:
            lines.append(f"| {n} | not run | | | | | |")
            continue
        s = summarize(m.preds)
        cost = f"${m.cost_per_1k_usd:.4f}" if m.cost_per_1k_usd is not None else "n/a"
        p95 = f"{m.p95_ms:.0f}" if m.p95_ms is not None else "n/a"
        gh = f"{m.train_gpu_h:.2f}" if m.train_gpu_h is not None else "n/a"
        lines.append(
            f"| {n} | {s['f1']:.3f} | {s['asr'] * 100:.1f}% | {s['fpr'] * 100:.2f}% | {p95} | {cost} | {gh} |"
        )
    return lines


def seeds_table(title: str, per_seed: dict[str, list[Pred]]) -> list[str]:
    """Per-seed metrics plus mean +- std (PRD 9.4)."""
    import numpy as np

    lines = [f"### {title}", "", "| seed | ASR | FPR | F1 | AUROC |", "|---|---|---|---|---|"]
    cols: dict[str, list[float]] = {k: [] for k in ("asr", "fpr", "f1", "auroc")}
    for seed, preds in per_seed.items():
        s = summarize(preds)
        for k in cols:
            cols[k].append(s[k])
        lines.append(
            f"| {seed} | {s['asr'] * 100:.2f}% | {s['fpr'] * 100:.2f}% | {s['f1']:.4f} | {s['auroc']:.4f} |"
        )
    if len(per_seed) > 1:
        f = {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in cols.items()}
        lines.append(
            f"| mean +- std (n={len(per_seed)}) | {f['asr'][0] * 100:.2f} +- {f['asr'][1] * 100:.2f}% | "
            f"{f['fpr'][0] * 100:.2f} +- {f['fpr'][1] * 100:.2f}% | {f['f1'][0]:.4f} +- {f['f1'][1]:.4f} | "
            f"{f['auroc'][0]:.4f} +- {f['auroc'][1]:.4f} |"
        )
    return lines


def retention_sentence(
    small: ModelEval | None,
    big: ModelEval | None,
    small_label: str = "0.6B",
    big_label: str = "1.7B",
) -> str:
    if small is None or big is None or small.p95_ms is None or big.p95_ms is None:
        return f"{small_label} vs {big_label} retention: not available (a required run or latency is missing)."
    f_s, f_b = summarize(small.preds)["f1"], summarize(big.preds)["f1"]
    keep = 100 * f_s / f_b if f_b else 0.0
    speed = big.p95_ms / small.p95_ms if small.p95_ms else 0.0
    return f"{small_label} keeps {keep:.1f}% of {big_label}'s F1 at {speed:.1f}x lower p95 latency."


def render_report(
    version: str,
    rows: list[Row],
    models: dict[str, ModelEval],
    *,
    baseline: str,
    focus: str,
    size_rows: list[str],
    leak_suite: list[dict[str, object]] | None = None,
    resamples: int = 1000,
    seed: int = 42,
    retention: tuple[str, str] = ("qwen3-0.6b-sft", "qwen3-1.7b-sft"),
    extra_sections: list[str] | None = None,
    title_suffix: str = "",
    retention_text: str | None = None,
    leak_notes: list[str] | None = None,
) -> str:
    out = [
        f"# Eval {version}{title_suffix}",
        "",
        f"gold_test rows: {len(rows)}; bootstrap resamples: {resamples}; seed: {seed}.",
        "",
    ]
    out += [
        "## Comparison (gold_test)",
        "",
        "| model | ASR [95% CI] | FPR [95% CI] | F1 [95% CI] | AUROC | hard-benign FPR |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in models.items():
        s = summarize(m.preds)
        out.append(
            f"| {name} | {ci_cell(m.preds, 'asr', resamples, seed)} | {ci_cell(m.preds, 'fpr', resamples, seed)} | "
            f"{ci_cell(m.preds, 'f1', resamples, seed)} | {s['auroc']:.3f} | {s['hard_benign_fpr'] * 100:.2f}% |"
        )
    base = models.get(baseline)
    foc = models.get(focus)
    out += ["", "## Significance (McNemar, paired on correctness)", ""]
    if base and foc:
        r = mcnemar(correct(foc.preds), correct(base.preds))
        verdict = "significant" if r.p_value < 0.05 else "not significant"
        rel = 0.0
        a0, a1 = asr(base.preds), asr(foc.preds)
        if a0:
            rel = (a0 - a1) / a0 * 100
        out.append(
            f"{focus} vs {baseline}: b={r.b}, c={r.c}, p={r.p_value:.4g} ({r.method}) - {verdict}. "
            f"Relative ASR reduction {rel:.1f}%."
        )
    else:
        out.append(f"{focus} or {baseline} missing; test not run.")
    out += ["", "## Confusion matrices", ""]
    for name, m in models.items():
        c = confusion(m.preds)
        out.append(f"- {name}: TP {c['tp']}, FP {c['fp']}, FN {c['fn']}, TN {c['tn']}")
    out += ["", "## Slices (ASR / FPR by slice)", ""]
    for name, m in models.items():
        w = worst_slice(m.preds)
        out.append(
            f"- {name}: worst slice = {w[0]}={w[1]} (ASR {w[2] * 100:.1f}%)"
            if w
            else f"- {name}: no slice has enough attacks"
        )
    if foc:
        out += [
            "",
            f"### {focus}: full slice table",
            "",
            "| slice | value | n | ASR | FPR |",
            "|---|---|---|---|---|",
        ]
        for sl, vals in slice_report(foc.preds).items():
            for v, mm in vals.items():
                out.append(
                    f"| {sl} | {v} | {int(mm['n'])} | {mm['asr'] * 100:.1f}% | {mm['fpr'] * 100:.2f}% |"
                )
        out += ["", f"## Worst 20 misses ({focus})", ""]
        for r_, p_ in worst_misses(rows, foc.preds):
            kind = "FN (attack passed)" if p_.y_true == 1 else "FP (benign blocked)"
            out.append(
                f"- {kind}, p_unsafe={p_.p_unsafe:.3f}, family={r_.attack_family}, lang={r_.language}: {r_.text[:200]!r}"
            )
    out += ["", "## Size vs quality", ""] + size_rows
    out += [
        "",
        retention_text or retention_sentence(models.get(retention[0]), models.get(retention[1])),
    ]
    out += extra_sections or []
    if leak_suite:
        out += [
            "",
            "## System leak suite (temp 0)",
            "",
            "| level | n | leaks | rate [95% CI] |",
            "|---|---|---|---|",
        ]
        for lr in leak_suite:
            if "rate" not in lr:  # level not run: say why instead of inventing a row
                out.append(f"| {lr['level']} | - | - | {lr.get('status', 'not run')} |")
                continue
            rate, lo, hi = (float(str(lr[k])) for k in ("rate", "ci_lo", "ci_hi"))
            out.append(
                f"| {lr['level']} | {lr['n']} | {lr['leaks']} | "
                f"{rate * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}] |"
            )
        out += ["", *(leak_notes or [])]
    return "\n".join(out) + "\n"


def scorecard(version: str, models: dict[str, ModelEval]) -> dict[str, object]:
    return {
        "version": version,
        "models": {
            n: {
                **summarize(m.preds),
                "p95_ms": m.p95_ms,
                "cost_per_1k_usd": m.cost_per_1k_usd,
                "note": m.note,
            }
            for n, m in models.items()
        },
    }


def write_report(
    version: str, md: str, card: dict[str, object], reports_dir: Path, web_public: Path
) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    web_public.mkdir(parents=True, exist_ok=True)
    md_path, js_path = reports_dir / f"eval-{version}.md", web_public / "scorecard.json"
    md_path.write_text(md, encoding="utf-8")
    js_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return md_path, js_path
