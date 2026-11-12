"""Weekly drift report (PRD 11): embed new attempts, cluster, report the share outside known clusters.

``embed`` is injected (a small sentence-embedding model on a CPU box in production, a stub in tests).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]


@dataclass(frozen=True)
class DriftReport:
    n_new: int
    share_outside_known: float
    new_clusters: list[tuple[int, float]]  # (size, share of all new points)
    flagged: bool


def centroids_and_radii(
    known: Arr, labels: NDArray[np.int64], quantile: float = 0.95
) -> tuple[Arr, Arr]:
    cents, radii = [], []
    for k in sorted(set(labels.tolist()) - {-1}):
        pts = known[labels == k]
        c = pts.mean(axis=0)
        cents.append(c)
        radii.append(float(np.quantile(np.linalg.norm(pts - c, axis=1), quantile)))
    return np.array(cents), np.array(radii)


def cluster_known(known: Arr, min_cluster_size: int = 10) -> NDArray[np.int64]:
    from sklearn.cluster import HDBSCAN

    return np.asarray(HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(known), dtype=np.int64)


def drift_report(
    known: Arr, new: Arr, *, min_cluster_size: int = 10, flag_share: float = 0.05
) -> DriftReport:
    from sklearn.cluster import HDBSCAN

    labels = cluster_known(known, min_cluster_size)
    cents, radii = centroids_and_radii(known, labels)
    d = np.linalg.norm(new[:, None, :] - cents[None, :, :], axis=2)
    inside = (d <= radii[None, :]).any(axis=1) if len(cents) else np.zeros(len(new), dtype=bool)
    outside = new[~inside]
    share = float(len(outside) / len(new)) if len(new) else 0.0
    clusters: list[tuple[int, float]] = []
    if len(outside) >= min_cluster_size:
        lab = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True).fit_predict(
            outside
        )
        for k in sorted(set(lab.tolist()) - {-1}):
            size = int((lab == k).sum())
            clusters.append((size, size / len(new)))
    return DriftReport(len(new), share, clusters, any(s > flag_share for _, s in clusters))


def build_report(
    known_texts: Sequence[str],
    new_texts: Sequence[str],
    embed: Callable[[Sequence[str]], Arr],
    week: str,
) -> tuple[DriftReport, str]:
    rep = drift_report(embed(known_texts), embed(new_texts))
    md = [
        f"# Drift report {week}",
        "",
        f"- new attempts: {rep.n_new}",
        f"- share outside known clusters: {rep.share_outside_known * 100:.1f}%",
        f"- new clusters: {', '.join(f'{n} rows ({s * 100:.1f}%)' for n, s in rep.new_clusters) or 'none'}",
        f"- flagged (a new cluster above 5%): {'YES' if rep.flagged else 'no'}",
    ]
    return rep, "\n".join(md) + "\n"
