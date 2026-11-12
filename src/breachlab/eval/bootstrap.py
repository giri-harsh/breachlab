"""Seeded bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from breachlab.eval.metrics import Pred


def bootstrap_ci(
    preds: list[Pred],
    metric: Callable[[list[Pred]], float],
    *,
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point, lo, hi) via the percentile bootstrap."""
    if not preds:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(preds)
    vals = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n, size=n)
        vals[b] = metric([preds[i] for i in idx])
    a = (1 - level) / 2
    return metric(preds), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
