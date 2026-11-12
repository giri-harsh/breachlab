"""Parity of exported ONNX variants vs the torch logits on the gold rows the export kernel scored.

Single-adapter export (`run_export`, DPO 0.6B):
    uv run python scripts/run_parity.py --export-dir data/eval/export2 \
        --torch data/eval/gold/gold_test_dpo-0.6b-s0.npy --out data/eval/export2/parity.json

Multi-model export (scripts/kernel_export_body.py, SFT 0.6B + 1.7B on one CPU):
    uv run python scripts/run_parity.py --export3 data/eval/exportA --out data/eval/exportA/summary.json

Reads `export.json`/`export3.json` (latency), `calibration.json` or the per-model results.json (T, threshold)
and the `gold300_*_logits.npy` files (first N gold rows). No model is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from breachlab.eval.parity import Parity, pick_winner, summarize_export3, variant_parity
from breachlab.train.scoring import read_rows

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data" / "eval"
# which -> (torch gold logits, results.json holding the SFT calibration)
TORCH = {
    "sft-0.6b-s0": (EV / "gold/gold_test_sft-0.6b-s0.npy", EV / "dpo_results.json"),
    "sft-1.7b-s0": (EV / "gold17/gold_test_sft-1.7b-s0.npy", EV / "s17/results.json"),
}


def run(export_dir: Path, torch_npy: Path, gold_jsonl: Path) -> dict[str, object]:
    exp = json.loads((export_dir / "export.json").read_text())
    cal = json.loads((export_dir / "calibration.json").read_text())
    rows = read_rows(gold_jsonl)
    torch = np.load(torch_npy)
    parities: dict[str, Parity] = {}
    for f in sorted(export_dir.glob("gold300_*_logits.npy")):
        variant = f.name.removeprefix("gold300_").removesuffix("_logits.npy")
        lg = np.load(f)
        n = lg.shape[0]
        y = [int(r.label == "UNSAFE") for r in rows[:n]]
        parities[variant] = variant_parity(y, torch[:n], lg, cal["temperature"], cal["threshold"])
    lat = exp.get("latency", {})
    p50 = {v: lat[f"{v}_tok512_c1"]["p50_ms"] for v in parities if f"{v}_tok512_c1" in lat}
    p95 = {v: lat[f"{v}_tok512_c1"]["p95_ms"] for v in parities if f"{v}_tok512_c1" in lat}
    return {
        "which": exp.get("which"),
        "calibration": cal,
        "torch_reference": str(torch_npy),
        "variants": {v: p.as_dict() for v, p in parities.items()},
        "latency_tok512_c1": {v: {"p50_ms": p50.get(v), "p95_ms": p95.get(v)} for v in parities},
        "winner": pick_winner(parities, p95, p50),
    }


def run_export3(export3_dir: Path, gold_jsonl: Path) -> dict[str, object]:
    exp = json.loads((export3_dir / "export3.json").read_text())
    rows = read_rows(gold_jsonl)
    y = [int(r.label == "UNSAFE") for r in rows]

    def variant_logits(which: str, variant: str) -> np.ndarray | None:
        f = export3_dir / f"gold300_{which}_{variant}_logits.npy"
        return np.load(f) if f.exists() else None

    def torch_logits(which: str) -> np.ndarray:
        return np.load(TORCH[which][0])

    def calibration(which: str) -> dict[str, float]:
        res = json.loads(TORCH[which][1].read_text())
        return dict(res["sft"]["calibration"])

    return summarize_export3(exp, variant_logits, torch_logits, calibration, y)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path)
    ap.add_argument("--export3", type=Path, help="dir with export3.json + gold300_*_logits.npy")
    ap.add_argument("--torch", type=Path)
    ap.add_argument("--gold", type=Path, default=ROOT / "data/processed/v1/gold_test.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.export3:
        res = run_export3(a.export3, a.gold)
    elif a.export_dir and a.torch:
        res = run(a.export_dir, a.torch, a.gold)
    else:
        ap.error("give --export3 DIR, or --export-dir DIR with --torch NPY")
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
