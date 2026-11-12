"""Verbalizer scoring: softmax over 4 answer-token logits divided by temperature (PRD 7.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row, jsonl_lines
from breachlab.train.prompts import LABELS

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class Verdict:
    label: str  # SAFE | UNSAFE
    category: str  # none | injection | jailbreak | exfiltration
    p_unsafe: float
    probs: tuple[float, float, float, float]


def probs_from_logits(logits: Arr, temperature: float = 1.0) -> Arr:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float64)


def decide(logits: Arr, temperature: float = 1.0, threshold: float = 0.5) -> Verdict:
    p = probs_from_logits(logits, temperature)
    p_unsafe = float(1.0 - p[0])
    if p_unsafe >= threshold:
        cat = LABELS[1 + int(np.argmax(p[1:]))]
        return Verdict("UNSAFE", cat, p_unsafe, (p[0], p[1], p[2], p[3]))
    return Verdict("SAFE", "none", p_unsafe, (p[0], p[1], p[2], p[3]))


def read_rows(path: Path) -> list[Row]:
    return [Row.model_validate(json.loads(x)) for x in jsonl_lines(path.read_text("utf-8"))]
