"""Stage 2: hard-negative mining on the held-out mining pool (PRD 7.4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row
from breachlab.train.prompts import LABELS, label_index
from breachlab.train.scoring import probs_from_logits

Kind = Literal["false_negative", "false_positive", "wrong_category", "near_miss"]


@dataclass(frozen=True)
class Mined:
    row: Row
    kind: Kind
    top: int  # model's top answer index
    margin: float  # p(correct) - best wrong p
    runner_up: int  # most likely wrong class


def mine(
    rows: list[Row], logits: NDArray[np.float64], temperature: float = 1.0, near_miss: float = 0.2
) -> list[Mined]:
    """Collect FNs, FPs, category errors and correct-but-low-margin rows."""
    if len(rows) != len(logits):
        raise ValueError("rows and logits differ in length")
    p = probs_from_logits(logits, temperature)
    out: list[Mined] = []
    for r, pr in zip(rows, p, strict=True):
        gold, top = label_index(r), int(np.argmax(pr))
        masked = pr.copy()
        masked[gold] = -1.0
        runner = int(np.argmax(masked))
        margin = float(pr[gold] - masked[runner])
        if top == gold:
            if margin < near_miss:
                out.append(Mined(r, "near_miss", top, margin, runner))
            continue
        if gold == 0:
            kind: Kind = "false_positive"
        elif top == 0:
            kind = "false_negative"
        else:
            kind = "wrong_category"
        out.append(Mined(r, kind, top, margin, runner))
    return out


def summary(mined: list[Mined]) -> dict[str, int]:
    s: dict[str, int] = {}
    for m in mined:
        s[m.kind] = s.get(m.kind, 0) + 1
    return s


def top_word(m: Mined) -> str:
    return LABELS[m.top]
