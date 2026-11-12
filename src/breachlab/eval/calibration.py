"""ECE (15 bins) and temperature scaling."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


def softmax(logits: Arr, temperature: float = 1.0) -> Arr:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=np.float64)


def ece(confidence: Arr, correct: Arr, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    n = len(confidence)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() / n * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def nll(logits: Arr, labels: NDArray[np.int64], temperature: float) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(
    logits: Arr, labels: NDArray[np.int64], lo: float = 0.05, hi: float = 10.0
) -> float:
    """Golden-section search for the T minimising validation NLL (convex in 1/T)."""
    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(80):
        if nll(logits, labels, c) < nll(logits, labels, d):
            b = d
        else:
            a = c
        c, d = b - phi * (b - a), a + phi * (b - a)
    return float((a + b) / 2)


def ece_from_logits(
    logits: Arr, labels: NDArray[np.int64], temperature: float = 1.0, bins: int = 15
) -> float:
    p = softmax(logits, temperature)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels).astype(np.float64)
    return ece(conf, correct, bins)
