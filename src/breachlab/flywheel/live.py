"""Turn labeled game attempts into dataset ``Row``s (shared by ``build_dataset`` and ``fresh_test``).

Trust rules: ``leak`` and ``human`` labels are used as they are; a ``judge`` label is used only when it was not
queued for review (confidence >= 0.7 and equal to the guard's call); unlabeled attempts are never used unless
``include_unjudged_benign`` opts into the PRD's "benign candidates". Canary and gold-review sessions never
appear. Messages are PII-scrubbed (``breachlab.data.pii``) before they become text. Rows from sessions whose
nickname starts with ``selfplay`` are flagged ``source="selfplay"``, everything else ``source="live"``.
All rows of one game session share ``parent_id = "session:<id>"`` so splits stay group-atomic.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt, GameSession, Label, Player
from breachlab.data.dedup import dedup
from breachlab.data.pii import scrub
from breachlab.data.schema import Row, jsonl_lines
from breachlab.flywheel.judge import (
    GOLD_REVIEW_PREFIX,
    REVIEW_CONFIDENCE,
    UNSAFE_CATEGORIES,
    needs_review,
)

SELFPLAY_PREFIX = "selfplay"
SPLIT_FILES = ("train", "val", "mining_pool", "gold_test")


def is_selfplay(nickname: str) -> bool:
    return nickname.strip().lower().startswith(SELFPLAY_PREFIX)


def group_of(session_id: str) -> str:
    return f"session:{session_id}"


@dataclass(frozen=True)
class LiveCandidate:
    row: Row
    session_id: str
    attempt_id: int
    created_at: datetime

    @property
    def group(self) -> str:
        return group_of(self.session_id)


@dataclass
class CollectStats:
    attempts_seen: int = 0
    excluded_canary_or_gold: int = 0
    unlabeled: int = 0
    pending_review: int = 0
    no_category: int = 0
    pii_scrubbed: int = 0
    unjudged_benign_used: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    benign_contamination_rate: float | None = (
        None  # share of judged "passed" attempts the judge calls UNSAFE
    )
    benign_judged: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_split_rows(directory: Path, splits: tuple[str, ...] = SPLIT_FILES) -> list[Row]:
    """Read ``<directory>/<split>.jsonl`` files that exist (the data-vN layout of ``breachlab.data.build``)."""
    rows: list[Row] = []
    for name in splits:
        f = directory / f"{name}.jsonl"
        if f.exists():
            rows.extend(Row.model_validate_json(x) for x in jsonl_lines(f.read_text("utf-8")))
    return rows


def _excluded_count(db: Session) -> int:
    return int(
        db.scalar(
            sa.select(sa.func.count())
            .select_from(Attempt)
            .join(GameSession, GameSession.id == Attempt.session_id)
            .where(
                sa.or_(
                    GameSession.is_canary.is_(True),
                    Attempt.session_id.startswith(GOLD_REVIEW_PREFIX),
                    Attempt.level < 1,
                )
            )
        )
        or 0
    )


def collect_candidates(
    db: Session,
    *,
    include_unjudged_benign: bool = False,
    review_threshold: float = REVIEW_CONFIDENCE,
    since: datetime | None = None,
) -> tuple[list[LiveCandidate], CollectStats]:
    """Trusted, PII-scrubbed candidates ordered oldest first, plus counters for the dataset stats."""
    stats = CollectStats(excluded_canary_or_gold=_excluded_count(db))
    q = (
        sa.select(Attempt, Label, Player.nickname)
        .join(GameSession, GameSession.id == Attempt.session_id)
        .join(Player, Player.id == GameSession.player_id)
        .outerjoin(Label, Label.attempt_id == Attempt.id)
        .where(
            GameSession.is_canary.is_(False),
            sa.not_(Attempt.session_id.startswith(GOLD_REVIEW_PREFIX)),
            Attempt.level >= 1,
        )
        .order_by(Attempt.created_at, Attempt.id)
    )
    if since is not None:
        q = q.where(Attempt.created_at >= since)
    out: list[LiveCandidate] = []
    judged_benign = unsafe_benign = 0
    for a, lab, nick in db.execute(q):
        stats.attempts_seen += 1
        if lab is not None and lab.source == "judge" and a.verdict != "UNSAFE" and not a.leaked:
            judged_benign += 1
            unsafe_benign += lab.label == "UNSAFE"
        label: str
        category: str | None
        origin: str
        if lab is None:
            stats.unlabeled += 1
            if not (include_unjudged_benign and not a.leaked and a.verdict != "UNSAFE"):
                continue
            label, category, origin = "SAFE", "none", "unjudged"
            stats.unjudged_benign_used += 1
        else:
            if needs_review(a, lab, review_threshold):
                stats.pending_review += 1
                continue
            label, category, origin = lab.label, lab.category or a.category, lab.source
        if label == "UNSAFE" and category not in UNSAFE_CATEGORIES:
            stats.no_category += 1
            continue
        text = scrub(a.message)
        if text != a.message:
            stats.pii_scrubbed += 1
        source = "selfplay" if is_selfplay(nick) else "live"
        row = Row.new(
            text,
            "UNSAFE" if label == "UNSAFE" else "SAFE",
            category if label == "UNSAFE" else "none",  # type: ignore[arg-type]
            source,
            attack_family=f"live_{origin}",
            parent_id=group_of(a.session_id),
        )
        out.append(LiveCandidate(row, a.session_id, a.id, a.created_at))
        stats.by_source[source] = stats.by_source.get(source, 0) + 1
    stats.benign_judged = judged_benign
    stats.benign_contamination_rate = unsafe_benign / judged_benign if judged_benign else None
    return out, stats


def dedup_candidates(cands: Sequence[LiveCandidate]) -> list[LiveCandidate]:
    """Exact then near-duplicate removal inside the live pool; the earliest candidate of a cluster wins."""
    seen: set[str] = set()
    uniq: list[LiveCandidate] = []
    for c in cands:
        if c.row.id not in seen:
            seen.add(c.row.id)
            uniq.append(c)
    kept = {r.id for r in dedup([c.row for c in uniq])}
    return [c for c in uniq if c.row.id in kept]


def label_counts(rows: list[Row]) -> dict[str, dict[str, int]]:
    return {
        "label": dict(Counter(r.label for r in rows)),
        "category": dict(Counter(r.category for r in rows)),
        "source": dict(Counter(r.source for r in rows)),
    }
