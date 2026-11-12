"""Stage 4: temperature scaling on val logits, ECE before/after, threshold at FPR<=2% (PRD 7.4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row
from breachlab.eval.calibration import ece_from_logits, fit_temperature
from breachlab.eval.metrics import Pred, threshold_for_fpr
from breachlab.train.prompts import LABELS, label_index
from breachlab.train.scoring import decide, probs_from_logits


@dataclass(frozen=True)
class Calibration:
    temperature: float
    threshold: float
    ece_before: float
    ece_after: float


def to_preds(
    rows: list[Row], logits: NDArray[np.float64], temperature: float, threshold: float
) -> list[Pred]:
    out: list[Pred] = []
    for r, lg in zip(rows, logits, strict=True):
        v = decide(lg, temperature, threshold)
        out.append(
            Pred(
                y_true=int(r.label == "UNSAFE"),
                y_pred=int(v.label == "UNSAFE"),
                p_unsafe=v.p_unsafe,
                cat_true=r.category,
                cat_pred=v.category,
                hard_benign=r.attack_family == "hard_benign",
                slices=(
                    ("attack_family", r.attack_family),
                    ("language", r.language),
                    ("encoding", r.encoding),
                    ("length_bucket", r.length_bucket),
                    ("source", r.source),
                ),
            )
        )
    return out


def calibrate(
    rows: list[Row], logits: NDArray[np.float64], bins: int = 15, max_fpr: float = 0.02
) -> Calibration:
    labels = np.array([label_index(r) for r in rows], dtype=np.int64)
    t = fit_temperature(logits, labels)
    before = ece_from_logits(logits, labels, 1.0, bins)
    after = ece_from_logits(logits, labels, t, bins)
    preds = to_preds(rows, logits, t, 0.5)
    return Calibration(t, threshold_for_fpr(preds, max_fpr), before, after)


def category_names() -> tuple[str, ...]:
    return LABELS


def p_unsafe(logits: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    return np.asarray(1.0 - probs_from_logits(logits, temperature)[:, 0], dtype=np.float64)
