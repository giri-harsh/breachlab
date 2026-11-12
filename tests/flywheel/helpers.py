"""Shared builders for flywheel tests: an in-memory database filled with players, sessions, attempts, labels."""

from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Attempt, Base, GameSession, Label, Player

T0 = datetime(2026, 10, 1, 12, 0, 0)
_ids = itertools.count(1)


def memory_db() -> Session:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return Session(engine)


def add_session(db: Session, nickname: str, *, canary: bool = False, sid: str | None = None) -> str:
    sid = sid or f"s{next(_ids):04d}"
    p = Player(nickname=nickname)
    db.add(p)
    db.flush()
    db.add(GameSession(id=sid, player_id=p.id, ip_hash="h", is_canary=canary, started_at=T0))
    db.flush()
    return sid


def add_attempt(
    db: Session,
    sid: str,
    message: str,
    *,
    level: int = 4,
    verdict: str | None = "SAFE",
    category: str | None = None,
    leaked: bool = False,
    minutes: int = 0,
    label: tuple[str, str | None, str, float | None] | None = None,
) -> int:
    """``label`` = (label, category, source, judge_confidence)."""
    a = Attempt(
        session_id=sid,
        level=level,
        message=message,
        verdict=verdict,
        category=category,
        confidence=0.5,
        leaked=leaked,
        created_at=T0 + timedelta(minutes=minutes),
    )
    db.add(a)
    db.flush()
    if label is not None:
        lab, cat, source, conf = label
        rationale = f"[{lab}|{cat or 'none'}] seeded" if source in {"judge", "human"} else None
        db.add(
            Label(
                attempt_id=a.id,
                label=lab,
                category=cat,
                source=source,
                judge_confidence=conf,
                judge_rationale=rationale,
            )
        )
        db.flush()
    return a.id


def scripted_judge(rules: dict[str, tuple[str, str, float]]):  # type: ignore[no-untyped-def]
    """MockLLM function: answers by substring match -> (label, category, confidence); default SAFE 0.9."""

    def fn(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS (JSON):\n", 1)[1])
        out = []
        for it in items:
            lab, cat, conf = "SAFE", "none", 0.9
            for needle, verdict in rules.items():
                if needle in it["text"]:
                    lab, cat, conf = verdict
            out.append(
                {
                    "id": it["id"],
                    "label": lab,
                    "category": cat,
                    "confidence": conf,
                    "rationale": "r",
                }
            )
        return "```json\n" + json.dumps(out) + "\n```"

    return fn


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
