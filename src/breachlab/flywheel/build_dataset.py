"""Flywheel dataset build (PRD 10 step 3): live attempts + labels -> PII scrub -> dedup -> fresh_test -> merge.

    uv run python -m breachlab.flywheel.build_dataset --existing data/processed --out data/live-v2 \\
        --fresh-n 300            # reads NEON_DATABASE_URL

Pipeline (see ``live.py`` for the trust rules):

1. collect trusted candidates (canary / gold-review sessions excluded, PII scrubbed, selfplay flagged);
2. dedup inside the live pool, then drop anything near-duplicating an existing split. Duplicates of ``gold_test``
   are counted separately (``dropped_gold_dup``): gold_test is frozen and must stay untouched;
3. carve ``fresh_test`` (the newest whole sessions), the rest is split into train/val by session so no parent
   group spans two splits;
4. validate with the schema and ``check_leakage`` over ``existing + new`` and verify fresh/train disjointness;
5. write ``live_train.jsonl``, ``live_val.jsonl``, ``fresh_test.jsonl`` and ``stats.json`` (nothing is written if a
   check fails).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.data.dedup import near_duplicates
from breachlab.data.leakage import check_leakage
from breachlab.data.schema import Row, validate_rows
from breachlab.flywheel.fresh_test import assert_fresh_disjoint, select_fresh
from breachlab.flywheel.judge import stable_fraction
from breachlab.flywheel.live import (
    LiveCandidate,
    collect_candidates,
    dedup_candidates,
    label_counts,
    load_split_rows,
)


@dataclass
class BuildResult:
    train: list[Row]
    val: list[Row]
    fresh_test: list[Row]
    stats: dict[str, Any]


def val_groups(groups: Sequence[str], val_frac: float) -> set[str]:
    """Deterministic ``val_frac`` share of the sessions (at least one when val_frac > 0 and there are two+)."""
    uniq = sorted(set(groups), key=lambda g: stable_fraction(f"val:{g}"))
    k = round(len(uniq) * val_frac)
    if val_frac > 0 and len(uniq) >= 2:
        k = max(k, 1)
    return set(uniq[:k])


def build_dataset(
    db: Session,
    existing: Sequence[Row],
    *,
    fresh_n: int = 0,
    val_frac: float = 0.10,
    include_unjudged_benign: bool = False,
    since: datetime | None = None,
) -> BuildResult:
    """Build the live-data increment. ``existing`` are the current splits (train/val/mining_pool/gold_test)."""
    cands, cstats = collect_candidates(
        db, include_unjudged_benign=include_unjudged_benign, since=since
    )
    n_collected = len(cands)
    cands = dedup_candidates(cands)
    stats: dict[str, Any] = {
        "collected": n_collected,
        "dropped_internal_dup": n_collected - len(cands),
    }

    gold = [r for r in existing if r.split == "gold_test"]
    gold_dups = set(near_duplicates(gold, [c.row for c in cands])) if gold else set()
    cands = [c for c in cands if c.row.id not in gold_dups]
    others = [r for r in existing if r.split != "gold_test"]
    other_dups = set(near_duplicates(others, [c.row for c in cands])) if others else set()
    cands = [c for c in cands if c.row.id not in other_dups]
    stats["dropped_gold_dup"] = len(gold_dups)
    stats["dropped_existing_dup"] = len(other_dups)

    fresh_c: list[LiveCandidate] = []
    rest = cands
    if fresh_n > 0:
        fresh_c, rest = select_fresh(cands, fresh_n)
    train: list[Row] = []
    val: list[Row] = []
    in_val = val_groups([c.group for c in rest], val_frac)
    for c in rest:
        if c.group in in_val:
            val.append(c.row.with_(split="val"))
        else:
            train.append(c.row.with_(split="train"))
    fresh = [c.row for c in fresh_c]  # split stays "unassigned" (frozen Row.split vocabulary)

    new_rows = [*train, *val, *fresh]
    if new_rows:
        validate_rows(new_rows)
    check_leakage([*existing, *train, *val])
    assert_fresh_disjoint(fresh, [*train, *val], existing)

    stats.update(
        {
            "collect": cstats.as_dict(),
            "train": len(train),
            "val": len(val),
            "fresh_test": len(fresh),
            "train_counts": label_counts(train),
            "val_counts": label_counts(val),
            "fresh_counts": label_counts(fresh),
            "sessions": {
                "train": len({r.parent_id for r in train}),
                "val": len({r.parent_id for r in val}),
                "fresh_test": len({r.parent_id for r in fresh}),
            },
            "selfplay_rows": sum(r.source == "selfplay" for r in new_rows),
            "existing_rows": len(existing),
        }
    )
    return BuildResult(train, val, fresh, stats)


def write_outputs(result: BuildResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("live_train", result.train),
        ("live_val", result.val),
        ("fresh_test", result.fresh_test),
    ):
        (out / f"{name}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8"
        )
    (out / "stats.json").write_text(
        json.dumps(result.stats, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs the production database
    from sqlalchemy.orm import sessionmaker

    from breachlab.common.dburl import normalize_db_url

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--existing",
        type=Path,
        required=True,
        help="dir with train/val/mining_pool/gold_test.jsonl",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh-n", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--include-unjudged-benign", action="store_true")
    args = ap.parse_args(argv)
    url = os.environ.get("NEON_DATABASE_URL")
    if not url:
        print("NEON_DATABASE_URL is not set", file=sys.stderr)
        return 2
    engine = sa.create_engine(normalize_db_url(url), pool_pre_ping=True)
    with sessionmaker(engine, expire_on_commit=False)() as db:
        res = build_dataset(
            db,
            load_split_rows(args.existing),
            fresh_n=args.fresh_n,
            val_frac=args.val_frac,
            include_unjudged_benign=args.include_unjudged_benign,
        )
    write_outputs(res, args.out)
    print(json.dumps({k: res.stats[k] for k in ("train", "val", "fresh_test", "dropped_gold_dup")}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
