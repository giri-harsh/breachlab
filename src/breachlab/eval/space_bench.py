"""Guard latency: the real Vercel measurement (ADR-0082) and the superseded 2-core Kaggle emulation ladder (ADR-0064).

The shipped host is a Vercel Python function with the int4 graph; its benchmark files are
``docs/lanes/B3-vercel-bench-{warm,cold}.json`` (``load_vercel_bench``). The rung ladder below it is kept as history.

Reads the ``bench.json`` files that ``breachlab.serve.bench_runner`` wrote on Kaggle CPU kernels
(``data/eval/space_bench/bench{2,3,4}/bench/bench.json``) and renders the ladder table. Pure functions, no model
calls. Numbers come from shared Kaggle hardware and are noisy; they are reported unrounded (two decimals).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNGS = ("fp32", "kv", "kv@256", "kv@128", "kv@96", "kv@64")
SHIP_RUNG = "kv"
# Which kernel run is the reference for a rung when more than one measured it (bench2 also measured kv).
PREFERRED_SOURCE = {
    "fp32": "bench2",
    "kv": "bench3",
    "kv@256": "bench3",
    "kv@128": "bench3",
    "kv@96": "bench4",
    "kv@64": "bench4",
    "int4": "vercel-warm",  # the real host (ADR-0082): the shipped measurement
}
QUALITY_KEYS = ("asr", "fpr", "f1", "auroc", "category_accuracy")
B2_PATH = "docs/lanes/B2-truncation-quality.json"
VERCEL_RUNG = "int4"
VERCEL_WARM = "docs/lanes/B3-vercel-bench-warm.json"
VERCEL_COLD = "docs/lanes/B3-vercel-bench-cold.json"


@dataclass(frozen=True)
class Rung:
    name: str
    source: str  # kernel run, e.g. "bench3"
    c1_p50_ms: float
    c1_p95_ms: float
    c4_p50_ms: float
    c4_p95_ms: float
    cold_start_s: float | None = None
    quality: dict[str, float] | None = None
    quality_n: int | None = None


def _rung(name: str, source: str, d: dict[str, Any]) -> Rung:
    return Rung(
        name=name,
        source=source,
        c1_p50_ms=float(d["c1"]["model_p50_ms"]),
        c1_p95_ms=float(d["c1"]["model_p95_ms"]),
        c4_p50_ms=float(d["c4"]["model_p50_ms"]),
        c4_p95_ms=float(d["c4"]["model_p95_ms"]),
        cold_start_s=float(d["cold_start_s"]) if "cold_start_s" in d else None,
        quality={k: float(v) for k, v in d["quality"].items()} if "quality" in d else None,
        quality_n=int(d["quality_n"]) if "quality_n" in d else None,
    )


def load_space_bench(root: Path) -> list[Rung]:
    """Every rung measurement found under ``root/bench*/bench/bench.json`` (a rung can appear in several runs)."""
    out: list[Rung] = []
    for f in sorted(root.glob("bench*/bench/bench.json")):
        source = f.parents[1].name
        data = json.loads(f.read_text(encoding="utf-8"))
        for name, d in data.items():
            if isinstance(d, dict) and "c1" in d and "c4" in d:
                out.append(_rung(name, source, d))
    return out


def pick(rungs: list[Rung], name: str) -> Rung | None:
    """The reference measurement of ``name``: the preferred kernel run, else the first one found."""
    cands = [r for r in rungs if r.name == name]
    pref = PREFERRED_SOURCE.get(name)
    return next((r for r in cands if r.source == pref), cands[0] if cands else None)


def repeats(rungs: list[Rung], name: str) -> list[Rung]:
    """Other measurements of the same rung (not the reference one): evidence of run-to-run noise."""
    ref = pick(rungs, name)
    return [r for r in rungs if r.name == name and r is not ref]


def ship_rung(rungs: list[Rung]) -> Rung | None:
    return pick(rungs, SHIP_RUNG)


def quality_delta(r: Rung, ref: Rung | None) -> tuple[bool, str]:
    """(worse than the full-length reference?, human-readable delta). Rates are fractions."""
    if r.quality is None:
        return False, "not measured"
    if ref is None or ref.quality is None or r is ref:
        return False, "reference"
    d = {
        k: r.quality[k] - ref.quality[k]
        for k in QUALITY_KEYS
        if k in r.quality and k in ref.quality
    }
    worse = (
        d.get("asr", 0) > 0
        or d.get("fpr", 0) > 0
        or d.get("f1", 0) < 0
        or d.get("category_accuracy", 0) < 0
    )
    parts = [
        f"ASR {d['asr'] * 100:+.2f} pp",
        f"FPR {d['fpr'] * 100:+.2f} pp",
        f"F1 {d['f1']:+.4f}",
        f"cat acc {d['category_accuracy'] * 100:+.2f} pp",
    ]
    return worse, "; ".join(parts)


def load_b2(root: Path) -> Any | None:
    """Lane B2's truncation-quality result, if it has been written."""
    p = root / B2_PATH
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ladder_table(rungs: list[Rung], history: bool = False) -> list[str]:
    ref = pick(rungs, SHIP_RUNG)
    lines = [
        "| rung | source | c1 p50 / p95 ms | c4 p50 / p95 ms | ASR | FPR | F1 | cat acc | measured quality loss vs kv (this run) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in RUNGS:
        r = pick(rungs, name)
        if r is None:
            lines.append(f"| {name} | not run | | | | | | | |")
            continue
        q = r.quality
        cells = (
            [
                f"{q['asr'] * 100:.2f}%",
                f"{q['fpr'] * 100:.2f}%",
                f"{q['f1']:.4f}",
                f"{q['category_accuracy'] * 100:.2f}%",
            ]
            if q
            else ["n/a"] * 4
        )
        worse, delta = quality_delta(r, ref)
        if q is None:
            loss = "not measured through the service (ONNX parity vs torch passed, ADR-0036)"
        elif r is ref:
            loss = "reference (prefix-KV is parity-safe: max logit diff 1.5e-05 vs fp32)"
        else:
            loss = f"{'YES' if worse else 'none'}: {delta}"
        mark = (
            (" (fp32 reference, superseded)" if history else " **(shipped)**")
            if name == SHIP_RUNG
            else ""
        )
        lines.append(
            f"| {name}{mark} | {r.source} | {r.c1_p50_ms:.2f} / {r.c1_p95_ms:.2f} | "
            f"{r.c4_p50_ms:.2f} / {r.c4_p95_ms:.2f} | {' | '.join(cells)} | {loss} |"
        )
    for name in RUNGS:
        for r in repeats(rungs, name):
            lines.append(
                f"| {name} (repeat) | {r.source} | {r.c1_p50_ms:.2f} / {r.c1_p95_ms:.2f} | "
                f"{r.c4_p50_ms:.2f} / {r.c4_p95_ms:.2f} | | | | | same config, second run: run-to-run noise |"
            )
    return lines


@dataclass(frozen=True)
class VercelBench:
    """The warm and cold benchmark of the shipped Vercel guard (docs/lanes/B3-vercel-bench-*.json)."""

    url: str
    started: str
    c1: dict[str, Any]
    c4: dict[str, Any]
    cold: list[dict[str, Any]]

    @property
    def rung(self) -> Rung:
        """The concurrency-1/4 model times as a ladder entry (source ``vercel-warm``)."""
        return Rung(
            name=VERCEL_RUNG,
            source=PREFERRED_SOURCE[VERCEL_RUNG],
            c1_p50_ms=float(self.c1["model_p50_ms"]),
            c1_p95_ms=float(self.c1["model_p95_ms"]),
            c4_p50_ms=float(self.c4["model_p50_ms"]),
            c4_p95_ms=float(self.c4["model_p95_ms"]),
            cold_start_s=self.first_request_s,
        )

    @property
    def first_request_s(self) -> float | None:
        """Round trip of the first request on a fresh instance, in seconds (None if no cold probe)."""
        cold = [c for c in self.cold if c.get("cold")]
        return float(cold[-1]["rtt_ms"]) / 1000 if cold else None


def load_vercel_bench(root: Path) -> VercelBench | None:
    """The Vercel benchmark under ``root`` (repository root), or None when the warm file is missing."""
    warm_f = root / VERCEL_WARM
    if not warm_f.exists():
        return None
    warm = json.loads(warm_f.read_text(encoding="utf-8"))
    cold_f = root / VERCEL_COLD
    cold = json.loads(cold_f.read_text(encoding="utf-8")) if cold_f.exists() else {}
    return VercelBench(
        url=str(warm["url"]),
        started=str(warm["started"]),
        c1=warm["warm"]["c1"],
        c4=warm["warm"]["c4"],
        cold=list(cold.get("cold", [])),
    )


def _errors(d: dict[str, Any]) -> str:
    e = d.get("errors_by_status", {})
    return ", ".join(f"{k} {v}" for k, v in e.items()) if e else "none"


def vercel_section(v: VercelBench, gate_max_ms: float | None, quality_note: str) -> list[str]:
    """Latency of the shipped host, stated plainly. ``quality_note`` is the int4 quality sentence (report-owned)."""
    c1, c4 = v.c1, v.c4
    out = [
        "## Latency",
        "",
        "**Shipped rung: int4 weight-only (q4b32: MatMulNBits, 4 bit, block 32, int8 embedding) ONNX with prefix-KV "
        f"reuse, on a Vercel Python function (1 vCPU, 2 GB, Hobby; ADR-0082). Guard model time at concurrency 1: "
        f"p50 {c1['model_p50_ms']:.2f} ms, p95 {c1['model_p95_ms']:.2f} ms** (warm, {c1['n']} requests). "
        "This p95 is the number in gate-input.json and scorecard.json. "
        f"**The 300 ms target is NOT met: the p95 is {c1['model_p95_ms'] / 300:.1f}x the target and even the p50 is "
        f"{c1['model_p50_ms'] / 300:.1f}x.**",
        "",
        f"Provenance: measured on the real function {v.url} from a client in the sandbox, {c1['n']} requests per "
        f"level, gold_test length distribution clipped to 500 characters like the gateway, started {v.started} UTC "
        "(docs/lanes/B3-vercel-bench-warm.json, cold probe in B3-vercel-bench-cold.json). Model time is the "
        "service's own `latency_ms`; round trip includes the network and Vercel's routing. This replaces the Kaggle "
        "2-core emulation of ADR-0064, which is kept below only as history.",
        "",
        "| scenario | requests | errors | round trip p50 / p95 / max ms | model time p50 / p95 ms | req/s |",
        "|---|---|---|---|---|---|",
    ]
    for label, d in (("warm, concurrency 1", c1), ("warm, concurrency 4", c4)):
        out.append(
            f"| {label} | {d['n']} | {_errors(d)} | {d['rtt_p50_ms']:.0f} / {d['rtt_p95_ms']:.0f} / "
            f"{d['rtt_max_ms']:.0f} | {d['model_p50_ms']:.2f} / {d['model_p95_ms']:.2f} | "
            f"{d['requests_per_s']:.2f} |"
        )
    out.append("")
    out += [
        "The single vCPU serializes work: concurrency 4 gives the same throughput as concurrency 1 "
        f"({c4['requests_per_s']:.2f} vs {c1['requests_per_s']:.2f} req/s) and about "
        f"{c4['rtt_p50_ms'] / c1['rtt_p50_ms']:.1f}x the round-trip p50 (requests queue), with no extra instance "
        "started. The model time p95 is dominated by the longest inputs and by the shared vCPU.",
        "",
    ]
    if v.cold:
        out += [
            "### Cold start",
            "",
            "| idle before request | instance | round trip ms | instance boot s | model ms |",
            "|---|---|---|---|---|",
        ]
        for c in v.cold:
            boot = f"{c['boot_s']:.1f}" if c.get("boot_s") is not None else "n/a"
            out.append(
                f"| {c['idle_s']} s | {'new (cold)' if c.get('cold') else 'warm (served ' + str(c.get('served')) + ')'} "
                f"| {c['rtt_ms']:.0f} | {boot} | {c['model_ms']:.0f} |"
            )
        out += [
            "",
            "An instance survives at least 2 minutes idle and is gone by 10 minutes. The first request after a long "
            "idle takes about 13 s (433 MB from the Hub, then the ONNX session load), longer than the gateway's "
            '`GUARD_TIMEOUT_S=8`, so that player sees the fail-closed 503 "guard waking up" (ADR-0050) and the next '
            "request is served. Cold starts are a documented limit, not fixed (no keep-warm, ADR-0082).",
            "",
        ]
    if gate_max_ms is not None:
        out += [
            f"Gate threshold in effect: p95 <= {gate_max_ms:.0f} ms (ADR-0082: measured service-side model p95 at "
            f"concurrency 1, {c1['model_p95_ms']:.1f} ms, times 1.2; same metric as ADR-0064). The 300 ms target "
            "is NOT met, so the latency check means 'within the relaxed v1 threshold'.",
            "",
        ]
    if quality_note:
        out += [quality_note, ""]
    return out


def latency_section(
    rungs: list[Rung],
    b2: Any | None,
    gate_max_ms: float | None = None,
    vercel: VercelBench | None = None,
    quality_note: str = "",
) -> list[str]:
    if vercel is not None:
        if not rungs:  # no Kaggle ladder available: the real host alone
            return vercel_section(vercel, gate_max_ms, quality_note)
        history = [
            "### Superseded: Kaggle 2-core emulation of the fp32 rungs (ADR-0039, ADR-0064)",
            "",
        ]
        ship = ship_rung(rungs)
        if ship is not None:
            history += [
                "Measured before the real host existed, on a shared Kaggle CPU kernel with the `space_app` pinned to 2 "
                f"of 4 cores. Not the shipped model (fp32, not int4): the fp32 prefix-KV rung measured c1 p95 "
                f"{ship.c1_p95_ms:.2f} ms there. Kept for the truncation-rung comparison, which is still the only "
                "evidence on prompt-length caps.",
                "",
            ]
        return (
            vercel_section(vercel, gate_max_ms, quality_note)
            + history
            + _ladder_block(rungs, b2, history=True)
        )
    ship = ship_rung(rungs)
    if ship is None:
        return ["## Latency", "", "CPU latency not measured.", ""]
    reps = [r.c1_p95_ms for r in rungs if r.name == SHIP_RUNG]
    out = [
        "## Latency",
        "",
        f"**Shipped rung: fp32 ONNX with prefix-KV reuse (`{ship.name}`). Guard model time at concurrency 1: "
        f"p50 {ship.c1_p50_ms:.2f} ms, p95 {ship.c1_p95_ms:.2f} ms** (500 requests, {ship.source}). "
        "This p95 is the number in gate-input.json and scorecard.json. **The 300 ms target is NOT met.**",
        "",
        "Provenance, stated plainly: there is no guard host (ADR-0063), so this was not measured on a Space. It was "
        "measured on a shared Kaggle CPU kernel with the real `space_app` under uvicorn pinned to 2 of 4 cores "
        "(a 2-core server emulation, ADR-0064), gold_test texts clipped to 500 characters like the gateway. "
        "The hardware is shared and noisy: the same prefix-KV config measured "
        f"c1 p95 {min(reps):.2f} to {max(reps):.2f} ms across {len(reps)} runs (see the repeat row). "
        "Re-measure on the final host. Times are model times (the service's own `ms` field), unrounded; "
        "round-trip times are in the raw bench.json files under data/eval/space_bench/.",
        "",
    ]
    if gate_max_ms is not None:
        out += [
            f"Gate threshold in effect: p95 <= {gate_max_ms:.0f} ms (ADR-0064: measured + 20%).",
            "",
        ]
    return out + _ladder_block(rungs, b2)


def _ladder_block(rungs: list[Rung], b2: Any | None, history: bool = False) -> list[str]:
    """The rung ladder, its caveats and the Lane B2 truncation row."""
    out: list[str] = []
    best = min(
        (r for r in (pick(rungs, n) for n in RUNGS) if r is not None), key=lambda r: r.c1_p95_ms
    )
    out += ["### Rung ladder (ADR-0039)", ""] + ladder_table(rungs, history=history) + [""]
    out += [
        f"No rung reaches a 300 ms p95 on this hardware; the fastest is `{best.name}` at {best.c1_p95_ms:.2f} ms.",
        "",
    ]
    out += [
        "Quality columns come from the same service path on all 1,500 gold_test rows (texts clipped to 500 characters, "
        "calibration of the exported adapter), so they differ slightly from the torch rows above (full text). "
        "Rungs below kv truncate the message and are NOT shipped: their lower latency is bought with the quality "
        "deltas shown, and only Lane B2's dedicated truncation study can approve one.",
        "",
        "| truncation quality (Lane B2) | status |",
        "|---|---|",
    ]
    if b2 is None:
        out.append(
            f"| {B2_PATH} | not available at report time (placeholder, fill when Lane B2 reports) |"
        )
    else:
        blob = json.dumps(b2, sort_keys=True)
        out.append(f"| {B2_PATH} | {'`' + blob + '`' if len(blob) <= 800 else 'see the file'} |")
    out.append("")
    return out
