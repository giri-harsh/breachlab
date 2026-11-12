"""v1.1 read-only extras (CONTRACTS.md 7.7 daily boss board, 7.8 stats). Owned by lane V; registered by ``create_app``.

Both routes only read the v1 models in ``breachlab.common.db`` (no columns added by later
migrations), exclude canary sessions, and never return prompts, replies or secrets. ``/api/stats``
returns no nicknames either. Timestamps in the database are naive UTC (``service.now_utc``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt, GameSession, Level, Player

DAILY_LEVEL = 6
DAILY_BOSS = "Daily Vault"
DAILY_MAX_MESSAGES = 10
BOARD_SIZE = 20
STATS_TTL_S = 30.0
BOSSES: dict[int, str] = {
    1: "Chatty Bot",
    2: "Word Cop",
    3: "The Watcher",
    4: "Trained Sentinel",
    5: "Double Lock",
    6: "Daily Vault",
}


class BoardEntry(BaseModel):
    rank: int
    nickname: str
    messages: int
    at: str


class DailyOut(BaseModel):
    date: str
    level: int = DAILY_LEVEL
    boss: str = DAILY_BOSS
    max_messages: int = DAILY_MAX_MESSAGES
    board: list[BoardEntry]


class LastBreach(BaseModel):
    boss: str
    at: str


class StatsOut(BaseModel):
    players_today: int
    messages_today: int
    bosses_defeated_today: int
    blocks_today: int
    last_breach: LastBreach | None
    updated_at: str


def utc_now() -> datetime:
    """Naive UTC, matching how the gateway stores timestamps."""
    return datetime.now(UTC).replace(tzinfo=None)


def iso(ts: datetime) -> str:
    """ISO-8601 with an explicit UTC offset so browsers parse it as UTC."""
    return ts.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day)
    return start, start + timedelta(days=1)


def daily_board(s: Session, day: date, limit: int = BOARD_SIZE) -> list[BoardEntry]:
    """Level-6 clears on ``day`` (UTC) by non-canary sessions: fewest messages, then earliest."""
    start, end = day_bounds(day)
    rows = s.execute(
        sa.select(Player.nickname, Level.messages_used, Level.cleared_at)
        .join(GameSession, GameSession.id == Level.session_id)
        .join(Player, Player.id == GameSession.player_id)
        .where(
            Level.level == DAILY_LEVEL,
            Level.cleared_at.is_not(None),
            Level.cleared_at >= start,
            Level.cleared_at < end,
            GameSession.is_canary.is_(False),
        )
        .order_by(Level.messages_used.asc(), Level.cleared_at.asc(), Level.session_id.asc())
        .limit(limit)
    ).all()
    return [
        BoardEntry(rank=i, nickname=nick, messages=int(msgs or 0), at=iso(at))
        for i, (nick, msgs, at) in enumerate(rows, start=1)
        if at is not None
    ]


def compute_stats(s: Session, now: datetime) -> StatsOut:
    start, end = day_bounds(now.date())
    real = GameSession.is_canary.is_(False)
    attempts_today = (
        sa.select(Attempt)
        .join(GameSession, GameSession.id == Attempt.session_id)
        .where(real, Attempt.created_at >= start, Attempt.created_at < end)
        .subquery()
    )
    messages = s.scalar(sa.select(sa.func.count()).select_from(attempts_today)) or 0
    blocks = (
        s.scalar(
            sa.select(sa.func.count())
            .select_from(attempts_today)
            .where(attempts_today.c.verdict == "UNSAFE")
        )
        or 0
    )
    active = sa.union(
        sa.select(GameSession.player_id).where(
            real, GameSession.started_at >= start, GameSession.started_at < end
        ),
        sa.select(GameSession.player_id)
        .join(Attempt, Attempt.session_id == GameSession.id)
        .where(real, Attempt.created_at >= start, Attempt.created_at < end),
    ).subquery()
    players = s.scalar(sa.select(sa.func.count()).select_from(active)) or 0
    cleared = (
        sa.select(Level.level, Level.cleared_at)
        .join(GameSession, GameSession.id == Level.session_id)
        .where(real, Level.cleared_at.is_not(None))
    )
    bosses = (
        s.scalar(
            sa.select(sa.func.count()).select_from(
                cleared.where(Level.cleared_at >= start, Level.cleared_at < end).subquery()
            )
        )
        or 0
    )
    last = s.execute(cleared.order_by(Level.cleared_at.desc()).limit(1)).first()
    last_breach = (
        LastBreach(boss=BOSSES.get(int(last[0]), f"Boss {int(last[0])}"), at=iso(last[1]))
        if last is not None and last[1] is not None
        else None
    )
    return StatsOut(
        players_today=int(players),
        messages_today=int(messages),
        bosses_defeated_today=int(bosses),
        blocks_today=int(blocks),
        last_breach=last_breach,
        updated_at=iso(now),
    )


class StatsCache:
    """Process-local cache: one recompute per ``ttl_s`` however many browsers poll."""

    def __init__(self, ttl_s: float = STATS_TTL_S, clock: Callable[[], float] = time.monotonic):
        self.ttl_s, self.clock = ttl_s, clock
        self._value: StatsOut | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    def get(self, compute: Callable[[], StatsOut]) -> StatsOut:
        with self._lock:
            t = self.clock()
            if self._value is None or t - self._at >= self.ttl_s:
                self._value, self._at = compute(), t
            return self._value


def build_router(
    db: Callable[[], Iterator[Session]],
    settings: Any = None,
    *,
    now: Callable[[], datetime] = utc_now,
    cache: StatsCache | None = None,
) -> APIRouter:
    """Routes added by lane V. ``db`` is the gateway's session dependency."""
    router = APIRouter()
    stats_cache = cache if cache is not None else StatsCache()

    @router.get("/api/daily", response_model=DailyOut)
    def daily(s: Session = Depends(db)) -> DailyOut:
        today = now().date()
        return DailyOut(date=today.isoformat(), board=daily_board(s, today))

    @router.get("/api/stats", response_model=StatsOut)
    def stats(s: Session = Depends(db)) -> StatsOut:
        return stats_cache.get(lambda: compute_stats(s, now()))

    return router
