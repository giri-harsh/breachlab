"""Sliding-window rate limits backed by the attempts table (works on stateless hosts)."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt, GameSession


def count_recent(
    db: Session, *, session_id: str | None, ip_hash: str | None, now: datetime
) -> tuple[int, int]:
    """(messages by this session in the last 60 s, messages from this IP hash in the last 60 s)."""
    since = now - timedelta(seconds=60)
    q = sa.select(sa.func.count()).select_from(Attempt).where(Attempt.created_at >= since)
    by_session = 0
    if session_id:
        by_session = db.scalar(q.where(Attempt.session_id == session_id)) or 0
    by_ip = 0
    if ip_hash:
        by_ip = (
            db.scalar(
                q.join(GameSession, GameSession.id == Attempt.session_id).where(
                    GameSession.ip_hash == ip_hash
                )
            )
            or 0
        )
    return int(by_session), int(by_ip)


def allowed(
    by_session: int, by_ip: int, per_session: int = 10, per_ip: int = 30
) -> tuple[bool, str]:
    if by_session >= per_session:
        return False, "slow down: 10 messages per minute per session"
    if by_ip >= per_ip:
        return False, "slow down: too many messages from your network"
    return True, "ok"


class SlidingWindow:
    """In-process sliding-window limiter for routes with no attempts row (hints, skips, admin).

    Each serverless instance keeps its own window (no new table, CONTRACTS 7 froze the schema), so with N warm
    instances a client can reach at most N x ``limit`` per window; on one instance it is exact.
    """

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Record one request for ``key``; False when it is over the limit (the request is still counted)."""
        now = self._clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            q.append(now)
            if len(self._hits) > 10_000:  # drop idle keys so memory stays bounded
                for k in [
                    k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s
                ]:
                    del self._hits[k]
            return len(q) <= self.limit
