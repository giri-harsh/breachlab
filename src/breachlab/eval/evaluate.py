"""Glue: rows + per-model verdicts -> aligned ``Pred`` lists (shared by every comparison row)."""

from __future__ import annotations

from collections.abc import Sequence

from breachlab.data.schema import Row
from breachlab.eval.metrics import Pred


def slices_of(r: Row) -> tuple[tuple[str, str], ...]:
    return (
        ("attack_family", r.attack_family),
        ("language", r.language),
        ("encoding", r.encoding),
        ("length_bucket", r.length_bucket),
        ("source", r.source),
    )


def make_preds(
    rows: Sequence[Row], labels: Sequence[str], cats: Sequence[str], p_unsafe: Sequence[float]
) -> list[Pred]:
    if not (len(rows) == len(labels) == len(cats) == len(p_unsafe)):
        raise ValueError("rows and predictions must be the same length")
    return [
        Pred(
            y_true=int(r.label == "UNSAFE"),
            y_pred=int(lab == "UNSAFE"),
            p_unsafe=float(p),
            cat_true=r.category,
            cat_pred=cat,
            hard_benign=r.attack_family == "hard_benign",
            slices=slices_of(r),
        )
        for r, lab, cat, p in zip(rows, labels, cats, p_unsafe, strict=True)
    ]


def threshold_labels(p_unsafe: Sequence[float], threshold: float) -> list[str]:
    return ["UNSAFE" if p >= threshold else "SAFE" for p in p_unsafe]
