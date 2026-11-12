"""Leakage checks: gold_test must not near-duplicate train; no parent group spans splits."""

from __future__ import annotations

from breachlab.data.dedup import near_duplicates
from breachlab.data.schema import Row
from breachlab.data.splits import root_ids


class LeakageError(AssertionError):
    pass


def check_leakage(rows: list[Row]) -> None:
    by_split: dict[str, list[Row]] = {}
    for r in rows:
        by_split.setdefault(r.split, []).append(r)
    roots = root_ids(rows)
    seen: dict[str, str] = {}
    for r in rows:
        prev = seen.setdefault(roots[r.id], r.split)
        if prev != r.split:
            raise LeakageError(f"parent group {roots[r.id]} spans splits {prev} and {r.split}")
    train = by_split.get("train", []) + by_split.get("val", [])
    for held in ("gold_test", "mining_pool"):
        dups = near_duplicates(train, by_split.get(held, []))
        if dups:
            raise LeakageError(f"{len(dups)} {held} rows near-duplicate train/val, e.g. {dups[0]}")
