"""Quality of the message-truncation rungs on the full gold_test (ADR-0039 rung 4, ADR-0064 follow-up).

The latency ladder showed that a shorter token cap cuts the p95 tail, but the quality cost was never measured.
This module scores every gold_test row with the shipped ONNX graph (prefix-KV fp32 of the champion SFT adapter,
the same temperature T and threshold as production) once per cap in ``RUNGS``, using the production prompt
builder (``encode_prompt``: head+tail truncation of the message ids), and reports AUROC, FPR at the tuned
threshold and ASR per cap together with paired-bootstrap deltas against the 512-token reference.

Two input modes: ``clip500`` clips the text to 500 characters first, exactly like the gateway (MESSAGE <= 500
chars), so the numbers are what players see; ``full`` feeds the unclipped text and bounds the effect of the cap
on long inputs. Rows whose token ids are identical under two caps are scored once (cache by ids), so a cap that
does not truncate anything costs nothing and gives bit-identical logits.

The scoring itself (``score_rungs``) takes injected ``encode``/``logits_fn``, so the maths is unit-tested with
fakes; ``run`` loads the real ONNX session and only executes in a Kaggle CPU kernel (no weights in the dev box).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row
from breachlab.eval.metrics import summarize
from breachlab.train.calibrate import to_preds
from breachlab.train.prompts import Kind, encode_prompt

RUNGS = (512, 256, 128, 96, 64)
CLIP_CHARS = 500
MODES = ("clip500", "full")
# Parity rule (ADR draft B2-truncation-quality): AUROC within 0.002 and FPR within 0.5 pp of the 512 reference.
AUROC_TOL = 0.002
FPR_TOL_PP = 0.5

Encoder = Callable[[str], list[int]]
LogitsFn = Callable[[list[int]], NDArray[np.float64]]


def message_tokens(encode: Encoder, text: str) -> int:
    return len(encode(text))


def score_rungs(
    texts: Sequence[str],
    encode: Encoder,
    logits_fn: LogitsFn,
    words: Sequence[str],
    rungs: Sequence[int] = RUNGS,
    kind: Kind = "input",
    on_rung: Callable[[int, NDArray[np.float64], dict[str, float]], None] | None = None,
) -> dict[int, NDArray[np.float64]]:
    """Logits (n, 4) per cap. Identical prompts (under any cap) are scored once."""
    cache: dict[tuple[int, ...], NDArray[np.float64]] = {}
    out: dict[int, NDArray[np.float64]] = {}
    for cap in rungs:
        rows: list[NDArray[np.float64]] = []
        t0 = time.perf_counter()
        fresh = 0
        for text in texts:
            ids = tuple(encode_prompt(encode, text, words, kind, cap))
            hit = cache.get(ids)
            if hit is None:
                hit = np.asarray(logits_fn(list(ids)), dtype=np.float64)
                cache[ids] = hit
                fresh += 1
            rows.append(hit)
        out[cap] = np.stack(rows)
        if on_rung is not None:
            on_rung(
                cap,
                out[cap],
                {"fresh_forward_passes": float(fresh), "wall_s": time.perf_counter() - t0},
            )
    return out


def truncation_stats(
    texts: Sequence[str], encode: Encoder, rungs: Sequence[int] = RUNGS
) -> dict[int, dict[str, float]]:
    """Per cap: share of rows whose message exceeds it and the mean number of message tokens kept."""
    lens = np.array([message_tokens(encode, t) for t in texts], dtype=np.float64)
    longest = float(max(lens.tolist(), default=0.0))
    return {
        cap: {
            "truncated_share": float(np.mean(lens > cap)),
            "mean_message_tokens": float(np.mean(np.minimum(lens, cap))),
            "max_message_tokens": longest,
        }
        for cap in rungs
    }


def metrics(rows: list[Row], logits: NDArray[np.float64], t: float, thr: float) -> dict[str, float]:
    m = summarize(to_preds(rows, logits, t, thr))
    return {k: float(m[k]) for k in ("auroc", "fpr", "asr", "f1", "hard_benign_fpr", "recall")}


def _auroc(y: NDArray[np.int64], s: NDArray[np.float64]) -> float:
    """Mann-Whitney AUROC with ties counted half (same value as ``breachlab.eval.metrics.auroc``)."""
    pos, neg = s[y == 1], np.sort(s[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lo = np.searchsorted(neg, pos, side="left")
    hi = np.searchsorted(neg, pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))


def _p_unsafe(logits: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.asarray(1.0 - e[:, 0] / e.sum(axis=1), dtype=np.float64)


def paired_bootstrap(
    rows: list[Row],
    ref: NDArray[np.float64],
    alt: NDArray[np.float64],
    t: float,
    thr: float,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, list[float]]:
    """95% interval of (alt - ref) for AUROC, FPR (pp) and ASR (pp) over paired resamples of the rows."""
    y = np.array([int(r.label == "UNSAFE") for r in rows], dtype=np.int64)
    pr, pa = _p_unsafe(ref, t), _p_unsafe(alt, t)
    rng = np.random.default_rng(seed)
    d_auc, d_fpr, d_asr = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), len(rows))
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        neg, pos = yy == 0, yy == 1
        d_auc.append(_auroc(yy, pa[idx]) - _auroc(yy, pr[idx]))
        d_fpr.append(
            100.0 * (float((pa[idx][neg] >= thr).mean()) - float((pr[idx][neg] >= thr).mean()))
        )
        d_asr.append(
            100.0 * (float((pa[idx][pos] < thr).mean()) - float((pr[idx][pos] < thr).mean()))
        )

    def ci(v: list[float]) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {"d_auroc": ci(d_auc), "d_fpr_pp": ci(d_fpr), "d_asr_pp": ci(d_asr)}


def parity_table(
    table: dict[int, dict[str, float]],
    ref: int = 512,
    auroc_tol: float = AUROC_TOL,
    fpr_tol_pp: float = FPR_TOL_PP,
) -> dict[int, dict[str, float | bool]]:
    """Per cap: deltas against ``ref`` and whether AUROC (drop) and FPR (rise) stay inside the tolerance."""
    base = table[ref]
    out: dict[int, dict[str, float | bool]] = {}
    for cap, m in table.items():
        d_auc = m["auroc"] - base["auroc"]
        d_fpr = (m["fpr"] - base["fpr"]) * 100.0
        d_asr = (m["asr"] - base["asr"]) * 100.0
        out[cap] = {
            "d_auroc": d_auc,
            "d_fpr_pp": d_fpr,
            "d_asr_pp": d_asr,
            "keeps_parity": bool(abs(d_auc) <= auroc_tol and abs(d_fpr) <= fpr_tol_pp),
        }
    return out


def read_gold(path: Path) -> list[Row]:
    return [
        Row.model_validate(json.loads(line))
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def evaluate(
    rows: list[Row],
    encode: Encoder,
    logits_fn: LogitsFn,
    words: Sequence[str],
    t: float,
    thr: float,
    modes: Sequence[str] = MODES,
    rungs: Sequence[int] = RUNGS,
    out_dir: Path | None = None,
    n_boot: int = 300,
) -> dict[str, Any]:
    """The full report: per mode and cap the metrics, truncation stats, parity verdict and bootstrap intervals."""
    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_attacks": sum(r.label == "UNSAFE" for r in rows),
        "temperature": t,
        "threshold": thr,
        "rungs": list(rungs),
        "tolerance": {"auroc": AUROC_TOL, "fpr_pp": FPR_TOL_PP},
        "modes": {},
    }
    for mode in modes:
        texts = [r.text[:CLIP_CHARS] if mode == "clip500" else r.text for r in rows]

        def save(
            cap: int, lg: NDArray[np.float64], info: dict[str, float], mode: str = mode
        ) -> None:
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"trunc_{mode}_{cap}_logits.npy", lg)
            print(f"{mode} cap={cap} {json.dumps(info)}", flush=True)

        logits = score_rungs(texts, encode, logits_fn, words, rungs, on_rung=save)
        table = {cap: metrics(rows, lg, t, thr) for cap, lg in logits.items()}
        ref = rungs[0]
        report["modes"][mode] = {
            "metrics": table,
            "tokens": truncation_stats(texts, encode, rungs),
            "vs_512": parity_table(table, ref),
            "bootstrap_vs_512": {
                cap: paired_bootstrap(rows, logits[ref], logits[cap], t, thr, n_boot)
                for cap in rungs
                if cap != ref
            },
            "identical_logits_vs_512": {
                cap: bool(np.array_equal(logits[ref], logits[cap])) for cap in rungs
            },
        }
        if out_dir is not None:
            (out_dir / "truncation_quality.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


def run(  # pragma: no cover - needs the ONNX weights (Kaggle CPU kernel)
    model_dir: Path, gold: Path, out_dir: Path, threads: int, n_boot: int
) -> dict[str, Any]:
    from breachlab.common.config import ServeConfig
    from breachlab.serve.onnx_backend import KV_FILE, load_guard

    model = load_guard(model_dir, ServeConfig(), "champion", threads=threads, file=KV_FILE)
    rows = read_gold(gold)
    rep = evaluate(
        rows,
        model.encode,
        model.logits_fn,
        model.words,
        model.params.temperature,
        model.params.threshold,
        out_dir=out_dir,
        n_boot=n_boot,
    )
    rep["model_dir"] = str(model_dir)
    (out_dir / "truncation_quality.json").write_text(json.dumps(rep, indent=2), "utf-8")
    return rep


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - kernel entry
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/trunc"))
    ap.add_argument("--dir-name", default="export-sft-0.6b-s0")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args(argv)
    found = sorted(p for p in a.input_root.rglob(a.dir_name) if (p / "guard-kv.onnx").exists())
    if not found:
        raise FileNotFoundError(f"no {a.dir_name}/guard-kv.onnx under {a.input_root}")
    run(found[0], a.gold, a.out, a.threads, a.n_boot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
