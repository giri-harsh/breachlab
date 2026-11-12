"""Exact + near-duplicate removal with MinHash LSH (Jaccard >= 0.8)."""

from __future__ import annotations

import re
from collections.abc import Iterable

from datasketch import MinHash, MinHashLSH

from breachlab.data.schema import Row

NUM_PERM = 128
THRESHOLD = 0.8
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.casefold()).strip()


def shingles(text: str, k: int = 3) -> set[str]:
    t = normalize(text)
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM, seed=1)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def dedup(rows: Iterable[Row], threshold: float = THRESHOLD) -> list[Row]:
    """Keep the first of each near-duplicate cluster (input order is the priority)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    seen_exact: set[str] = set()
    kept: list[Row] = []
    for row in rows:
        key = normalize(row.text)
        if key in seen_exact:
            continue
        mh = minhash(row.text)
        if lsh.query(mh):
            continue
        seen_exact.add(key)
        lsh.insert(row.id, mh)
        kept.append(row)
    return kept


def near_duplicates(a: Iterable[Row], b: Iterable[Row], threshold: float = THRESHOLD) -> list[str]:
    """IDs of rows in b that near-duplicate any row in a."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for row in a:
        lsh.insert(row.id, minhash(row.text))
    return [row.id for row in b if lsh.query(minhash(row.text))]
