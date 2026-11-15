"""Kaggle CPU kernel: score splits with the served int8 students through ``StudentScorer`` (the serving code path).

Also verifies tokenizer parity: the ``tokenizers`` encoding used at serving must give the same logits as the HF
tokenizer used in training (``logits_<split>_int8.npy`` of the distillation kernel), and records single-thread latency.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - Kaggle CPU
    from breachlab.serve.cascade import StudentScorer
    from breachlab.train.distill import read_rows

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidates", default="distilbert-ml,minilm-ml")
    ap.add_argument("--splits", default="val,gold_test,opener_slice_v1,ood_test")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict[str, object] = {}
    for cand in a.candidates.split(","):
        onnx = glob.glob(f"/kaggle/input/**/{cand}/student-int8.onnx", recursive=True)[0]
        d = Path(onnx).parent
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        scorer = StudentScorer(onnx, str(d / "tokenizer.json"), meta, threads=1)
        info: dict[str, object] = {}
        for split in a.splits.split(","):
            p = a.data_dir / f"{split}.jsonl"
            if not p.exists():
                continue
            rows = read_rows(p)
            t0 = time.perf_counter()
            lat = []
            out = np.zeros((len(rows), 4))
            for i, r in enumerate(rows):
                t1 = time.perf_counter()
                out[i] = scorer(r.text)
                lat.append((time.perf_counter() - t1) * 1000)
            np.save(a.out / f"{cand}_{split}_served.npy", out)
            entry: dict[str, object] = {
                "n": len(rows),
                "ms_p50": float(np.percentile(lat, 50)),
                "ms_p95": float(np.percentile(lat, 95)),
                "total_s": time.perf_counter() - t0,
            }
            ref = glob.glob(f"/kaggle/input/**/{cand}/logits_{split}_int8.npy", recursive=True)
            if ref:
                r0 = np.load(ref[0])
                entry["parity_vs_hf_tokenizer"] = {
                    "max_abs_logit_diff": float(np.abs(r0 - out).max()),
                    "argmax_agreement": float((r0.argmax(-1) == out.argmax(-1)).mean()),
                }
            info[split] = entry
            print(cand, split, entry, flush=True)
        res[cand] = info
    (a.out / "results.json").write_text(json.dumps(res, indent=1), "utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
