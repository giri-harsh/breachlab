"""Latency harness (PRD 7.3): N requests sampled from the gold_test length distribution, at fixed concurrency.

Run it on the same CPU image as production; ``score`` is any callable that performs one guard call.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    n: int
    concurrency: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    wall_s: float


def sample_texts(texts: Sequence[str], n: int, seed: int = 42) -> list[str]:
    """Sample with replacement so the length distribution of the source set is preserved."""
    rng = random.Random(seed)
    return [rng.choice(texts) for _ in range(n)]


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_latency(
    score: Callable[[str], object],
    texts: Sequence[str],
    *,
    requests: int = 500,
    concurrency: int = 1,
    warmup: int = 10,
    seed: int = 42,
) -> LatencyStats:
    batch = sample_texts(texts, requests, seed)
    for t in batch[:warmup]:
        score(t)

    def timed(t: str) -> float:
        t0 = time.perf_counter()
        score(t)
        return (time.perf_counter() - t0) * 1000

    w0 = time.perf_counter()
    if concurrency == 1:
        lat = [timed(t) for t in batch]
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            lat = list(ex.map(timed, batch))
    wall = time.perf_counter() - w0
    return LatencyStats(
        requests, concurrency, percentile(lat, 50), percentile(lat, 95), float(np.mean(lat)), wall
    )
