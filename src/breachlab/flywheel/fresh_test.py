"""fresh_test (PRD 8.3, 9.2): the newest held-out live attempts, disjoint from the training set by parent group.

A live row's parent group is its game session (``parent_id = "session:<id>"``): a player iterating on one attack
produces near-variants, so a session is either entirely in ``fresh_test`` or entirely in training, never both.
``fresh_test`` rows keep ``split="unassigned"`` because the frozen ``Row.split`` vocabulary has no fresh slot; they
live in their own file (``fresh_test.jsonl``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from sqlalchemy.orm import Session

from breachlab.data.dedup import near_duplicates
from breachlab.data.leakage import LeakageError
from breachlab.data.schema import Row
from breachlab.data.splits import root_ids
from breachlab.flywheel.live import (
    CollectStats,
    LiveCandidate,
    collect_candidates,
    dedup_candidates,
)


def select_fresh(
    cands: Sequence[LiveCandidate], n: int, exclude_groups: Iterable[str] = ()
) -> tuple[list[LiveCandidate], list[LiveCandidate]]:
    """Split into (fresh, rest): newest whole sessions until ``n`` rows are covered.

    Sessions are atomic, so the result can exceed ``n`` by less than one session. Groups in ``exclude_groups``
    (sessions already used for training) are never fresh; they go to ``rest`` untouched.
    """
    banned = set(exclude_groups)
    latest: dict[str, tuple[datetime, int]] = {}
    for c in cands:
        if c.group not in banned:
            latest[c.group] = max(
                latest.get(c.group, (c.created_at, c.attempt_id)), (c.created_at, c.attempt_id)
            )
    order = sorted(latest, key=lambda g: latest[g], reverse=True)
    chosen: set[str] = set()
    count = 0
    sizes: dict[str, int] = {}
    for c in cands:
        sizes[c.group] = sizes.get(c.group, 0) + 1
    for g in order:
        if count >= n:
            break
        chosen.add(g)
        count += sizes[g]
    fresh = [c for c in cands if c.group in chosen]
    rest = [c for c in cands if c.group not in chosen]
    return fresh, rest


def assert_fresh_disjoint(
    fresh: Sequence[Row], train: Sequence[Row], reference: Sequence[Row] = ()
) -> None:
    """Raise ``LeakageError`` if fresh shares a parent group with train, or near-duplicates train/reference."""
    all_rows = [*train, *fresh]
    roots = root_ids(all_rows)
    train_groups = {roots[r.id] for r in train}
    shared = {roots[r.id] for r in fresh} & train_groups
    if shared:
        raise LeakageError(
            f"{len(shared)} parent group(s) in both fresh_test and training, e.g. {sorted(shared)[0]}"
        )
    for name, rows in (("training", train), ("reference", reference)):
        dups = near_duplicates(rows, fresh) if rows else []
        if dups:
            raise LeakageError(
                f"{len(dups)} fresh_test rows near-duplicate {name} rows, e.g. {dups[0]}"
            )


def build_fresh_test(
    db: Session,
    train_rows: Sequence[Row],
    n: int,
    *,
    reference_rows: Sequence[Row] = (),
    since: datetime | None = None,
) -> tuple[list[Row], CollectStats]:
    """Newest trusted live rows (at most one session over ``n``) that the given training set has not seen.

    Excludes every session already in ``train_rows`` (by ``parent_id``), anything that near-duplicates
    ``train_rows`` or ``reference_rows`` (the existing splits including gold_test), then verifies disjointness.
    """
    cands, stats = collect_candidates(db, since=since)
    seen_groups = {r.parent_id for r in train_rows if r.parent_id}
    pool = [c for c in cands if c.group not in seen_groups]
    blocked = (
        set(near_duplicates([*train_rows, *reference_rows], [c.row for c in pool]))
        if (train_rows or reference_rows)
        else set()
    )
    pool = dedup_candidates([c for c in pool if c.row.id not in blocked])
    fresh, _ = select_fresh(pool, n)
    rows = [c.row for c in fresh]
    assert_fresh_disjoint(rows, train_rows, reference_rows)
    return rows, stats
