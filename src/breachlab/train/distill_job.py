"""Kaggle kernel entry for the distillation run (``python -m breachlab.train.distill_job``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from breachlab.train.distill import CANDIDATES, read_rows, teacher_logits, train_student


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapters", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--teacher-base", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--candidates", default="minilm-ml,distilbert-ml,xlmr-base")
    ap.add_argument(
        "--extra-splits",
        default="mining_pool,opener_slice_v1,attacks_tune,attacks_suite,attacks_fresh,ood_test",
    )
    ap.add_argument("--train-splits", default="train,mining_pool")
    ap.add_argument("--kd-temp", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--attack-weight", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    splits = {s: read_rows(a.data_dir / f"{s}.jsonl") for s in ("train", "val", "gold_test")}
    for s in a.extra_splits.split(","):
        p = a.data_dir / f"{s}.jsonl"
        if p.exists():
            splits[s] = read_rows(p)
    adapter = next(
        p for p in sorted(a.adapters.rglob("adapter")) if (p / "adapter_config.json").exists()
    )
    teacher = teacher_logits(adapter, a.teacher_base, splits, a.out)
    ids = {k: [r.id for r in v] for k, v in splits.items()}
    (a.out / "split_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    results = {}
    for name in a.candidates.split(","):
        try:
            results[name] = train_student(
                CANDIDATES[name],
                splits,
                teacher,
                a.out,
                epochs=a.epochs,
                train_names=tuple(a.train_splits.split(",")),
                kd_temp=a.kd_temp,
                alpha=a.alpha,
                attack_weight=a.attack_weight,
            )
        except Exception as exc:  # noqa: BLE001 - one failing candidate must not lose the others
            results[name] = {"error": repr(exc)[:400]}
            print("candidate failed:", name, repr(exc)[:400], flush=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=str), "utf-8")
    np.save(a.out / "ok.npy", np.zeros(1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
