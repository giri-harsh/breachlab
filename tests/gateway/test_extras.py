"""Lane V extras: /api/daily board (CONTRACTS 7.7) and /api/stats (7.8) on sqlite."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Attempt, Base, GameSession, Level, Player
from breachlab.gateway.extras import StatsCache, build_router, iso, utc_now

NOW = datetime(2026, 11, 12, 15, 30, 0)


@pytest.fixture
def factory() -> sessionmaker[Session]:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return sessionmaker(engine)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def client(factory: sessionmaker[Session], cache: StatsCache | None = None) -> TestClient:
    def db() -> Iterator[Session]:
        with factory() as s:
            yield s

    app = FastAPI()
    app.include_router(build_router(db, None, now=lambda: NOW, cache=cache))
    return TestClient(app)


def player(
    s: Session, sid: str, nick: str, canary: bool = False, started: datetime = NOW
) -> GameSession:
    p = Player(nickname=nick)
    s.add(p)
    s.flush()
    gs = GameSession(id=sid, player_id=p.id, ip_hash="h", is_canary=canary, started_at=started)
    s.add(gs)
    s.flush()
    return gs


def clear(s: Session, sid: str, level: int, msgs: int, at: datetime | None) -> None:
    s.add(Level(session_id=sid, level=level, secret_hash="x", messages_used=msgs, cleared_at=at))


def attempt(s: Session, sid: str, level: int, at: datetime, verdict: str | None = None) -> None:
    s.add(
        Attempt(
            session_id=sid,
            level=level,
            message="SECRET PROMPT",
            verdict=verdict,
            agent_reply="hidden reply",
            created_at=at,
        )
    )


def test_daily_board_order_filters_and_shape(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        player(s, "a", "Alice")
        player(s, "b", "Bob")
        player(s, "c", "Cara")
        player(s, "d", "Canary", canary=True)
        player(s, "e", "Yesterday")
        player(s, "f", "Unfinished")
        player(s, "g", "OtherLevel")
        clear(s, "a", 6, 4, NOW - timedelta(hours=2))
        clear(s, "b", 6, 2, NOW - timedelta(hours=1))
        clear(s, "c", 6, 4, NOW - timedelta(hours=3))  # same messages as Alice, earlier
        clear(s, "d", 6, 1, NOW - timedelta(hours=1))  # canary excluded
        clear(s, "e", 6, 1, NOW - timedelta(days=1))  # previous UTC day
        clear(s, "f", 6, 3, None)  # not cleared
        clear(s, "g", 5, 1, NOW - timedelta(hours=1))  # not the daily level
        s.commit()
    r = client(factory).get("/api/daily")
    assert r.status_code == 200
    body = r.json()
    assert body["date"] == "2026-11-12"
    assert (body["level"], body["boss"], body["max_messages"]) == (6, "Daily Vault", 10)
    assert [(e["rank"], e["nickname"], e["messages"]) for e in body["board"]] == [
        (1, "Bob", 2),
        (2, "Cara", 4),
        (3, "Alice", 4),
    ]
    assert body["board"][0]["at"] == "2026-11-12T14:30:00Z"


def test_daily_board_top_20_and_empty(factory: sessionmaker[Session]) -> None:
    assert client(factory).get("/api/daily").json()["board"] == []
    with factory() as s:
        for i in range(25):
            player(s, f"s{i}", f"P{i}")
            clear(s, f"s{i}", 6, 1 + i % 10, NOW - timedelta(minutes=i))
        s.commit()
    board = client(factory).get("/api/daily").json()["board"]
    assert len(board) == 20 and [e["rank"] for e in board] == list(range(1, 21))
    msgs = [e["messages"] for e in board]
    assert msgs == sorted(msgs)


def test_stats_counts_exclude_canary_and_leak_nothing(factory: sessionmaker[Session]) -> None:
    with factory() as s:
        player(s, "a", "Alice")
        player(s, "b", "Bob", started=NOW - timedelta(days=2))  # old session, plays today
        player(s, "c", "Cara", started=NOW - timedelta(days=2))  # idle today
        player(s, "k", "Canary", canary=True)
        attempt(s, "a", 1, NOW - timedelta(hours=1))
        attempt(s, "a", 2, NOW - timedelta(hours=1), verdict="UNSAFE")
        attempt(s, "b", 4, NOW - timedelta(minutes=5), verdict="UNSAFE")
        attempt(s, "b", 4, NOW - timedelta(minutes=4), verdict="SAFE")
        attempt(s, "c", 1, NOW - timedelta(days=1))  # yesterday
        attempt(s, "k", 1, NOW, verdict="UNSAFE")  # canary
        clear(s, "a", 1, 1, NOW - timedelta(hours=1))
        clear(s, "b", 3, 2, NOW - timedelta(minutes=10))
        clear(s, "c", 1, 1, NOW - timedelta(days=1))  # yesterday: not today, not latest
        clear(s, "k", 5, 1, NOW)  # canary: never the last breach
        s.commit()
    r = client(factory).get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "players_today": 2,
        "messages_today": 4,
        "bosses_defeated_today": 2,
        "blocks_today": 2,
        "last_breach": {"boss": "The Watcher", "at": "2026-11-12T15:20:00Z"},
        "updated_at": "2026-11-12T15:30:00Z",
    }
    text = r.text
    for banned in ("Alice", "Bob", "Canary", "SECRET PROMPT", "hidden reply"):
        assert banned not in text


def test_stats_empty_has_null_last_breach(factory: sessionmaker[Session]) -> None:
    body = client(factory).get("/api/stats").json()
    assert body["last_breach"] is None
    assert body["players_today"] == body["messages_today"] == 0


def test_stats_cached_for_ttl(factory: sessionmaker[Session]) -> None:
    clock = Clock()
    c = client(factory, StatsCache(ttl_s=30, clock=clock))
    assert c.get("/api/stats").json()["players_today"] == 0
    with factory() as s:
        player(s, "a", "Alice")
        s.commit()
    clock.t = 29.0
    assert c.get("/api/stats").json()["players_today"] == 0  # still cached
    clock.t = 30.0
    assert c.get("/api/stats").json()["players_today"] == 1


def test_helpers() -> None:
    assert iso(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05Z"
    assert utc_now().tzinfo is None


def test_registered_in_gateway_app() -> None:
    from pathlib import Path

    from breachlab.gateway import app as app_module

    assert "extras.build_router" in Path(app_module.__file__).read_text(encoding="utf-8")
