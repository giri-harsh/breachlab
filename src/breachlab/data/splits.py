"""Parent-grouped splits: all descendants of a root go to the same split (PRD 8.2/8.3)."""

from __future__ import annotations

import random
from collections import defaultdict

from breachlab.data.schema import Row


def root_ids(rows: list[Row]) -> dict[str, str]:
    """Map row id -> root ancestor id (follows parent_id chains; unknown parents act as roots)."""
    parent = {r.id: r.parent_id for r in rows}
    cache: dict[str, str] = {}

    def find(i: str) -> str:
        path = []
        cur = i
        while cur not in cache:
            p = parent.get(cur)
            if p is None or p == cur:
                cache[cur] = cur
                break
            path.append(cur)
            cur = p
        root = cache[cur]
        for n in path:
            cache[n] = root
        return root

    return {r.id: find(r.id) for r in rows}


def assign_splits(
    rows: list[Row],
    *,
    gold_n: int = 1500,
    mining_n: int = 5000,
    val_frac: float = 0.10,
    seed: int = 42,
) -> list[Row]:
    """Assign gold_test, then mining_pool, then val/train (90/10). Group-atomic, seeded."""
    roots = root_ids(rows)
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[roots[r.id]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    out: dict[str, str] = {}
    filled = {"gold_test": 0, "mining_pool": 0}
    targets = {"gold_test": gold_n, "mining_pool": mining_n}
    rest: list[str] = []
    for k in keys:
        size = len(groups[k])
        for name in ("gold_test", "mining_pool"):
            if filled[name] < targets[name]:
                filled[name] += size
                for r in groups[k]:
                    out[r.id] = name
                break
        else:
            rest.append(k)
    n_rest = sum(len(groups[k]) for k in rest)
    val_target = int(n_rest * val_frac)
    val_count = 0
    for k in rest:
        name = "val" if val_count < val_target else "train"
        if name == "val":
            val_count += len(groups[k])
        for r in groups[k]:
            out[r.id] = name
    return [r.with_(split=out[r.id]) for r in rows]
