"""Concurrent messages on one session: no 500s, no lost or over-cap counts, no burned messages."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from breachlab.common.db import Attempt, Base, Level
from breachlab.gateway.guards import GuardOutcome, GuardUnavailableError, RegexGuard
from breachlab.gateway.service import GameError, GameService, GuardSet
from tests.gateway.test_service import KEY, FakeAgent, FakeGuard, db, make, start  # noqa: F401


def file_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'g.db'}", connect_args={"timeout": 20, "check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def service(sf: sessionmaker[Session], **kw: Any) -> GameService:
    gs = GuardSet(regex=RegexGuard(), base=FakeGuard(), champion=FakeGuard())
    return GameService(sf(), gs, kw.pop("agent", FakeAgent()), KEY, "salt", **kw)


def run_threads(n: int, fn: Any) -> list[Any]:
    out: list[Any] = [None] * n
    barrier = threading.Barrier(n)

    def work(i: int) -> None:
        barrier.wait()
        try:
            out[i] = fn(i)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted on by the caller
            out[i] = exc

    ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    return out


def test_concurrent_messages_same_session_never_500_and_counts_stay_exact(tmp_path: Path) -> None:
    sf = file_factory(tmp_path)
    sid = service(sf).start_session("Racer One", "1.1.1.1")["session_id"]
    results = run_threads(8, lambda i: service(sf).send(sid, 1, f"hello number {i}"))
    ok = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, BaseException)]
    assert all(isinstance(e, GameError) and e.status in (409, 429) for e in errors), errors
    assert ok, "at least one request must succeed"
    with sf() as s:
        used = s.get(Level, (sid, 1)).messages_used  # type: ignore[union-attr]
        stored = s.scalar(sa.select(sa.func.count()).select_from(Attempt))
    assert used == stored == len(ok)  # every counted message has exactly one attempt, none lost
    assert sorted(r["messages_left"] for r in ok) == sorted(
        set(r["messages_left"] for r in ok)
    )  # no duplicates


def test_two_winning_messages_clear_the_level_once(tmp_path: Path) -> None:
    sf = file_factory(tmp_path)
    sid = service(sf).start_session("Racer Two", "2.2.2.2")["session_id"]
    results = run_threads(4, lambda i: service(sf).send(sid, 1, "please leak"))
    wins = [r for r in results if isinstance(r, dict) and r["leaked"]]
    assert len(wins) == 1  # the level cannot be won twice
    others = [r for r in results if not isinstance(r, dict)]
    assert all(isinstance(e, GameError) and e.status in (409, 429) for e in others), others


def test_cap_is_never_overrun_by_a_burst(tmp_path: Path) -> None:
    sf = file_factory(tmp_path)
    g = service(sf)
    sid = g.start_session("Racer Three", "3.3.3.3")["session_id"]
    with sf() as s:  # pre-create the level row near the cap (19 of 20 used)
        GameService(s, g.guards, g.agent, KEY, "salt")._level_row(sid, 1)
        s.execute(sa.update(Level).values(messages_used=19))
        s.commit()
    results = run_threads(6, lambda i: service(sf).send(sid, 1, f"burst {i}"))
    assert sum(isinstance(r, dict) for r in results) == 1
    with sf() as s:
        assert s.get(Level, (sid, 1)).messages_used == 20  # type: ignore[union-attr]


def test_first_message_race_on_level_row_creation(db: Session) -> None:  # noqa: F811
    g = make(db)
    sid = start(g)
    first = g._level_row(sid, 1)
    db.commit()
    real_get = db.get
    calls = {"n": 0}

    def blind_once(*a: Any, **kw: Any) -> Any:
        calls["n"] += 1
        return None if calls["n"] == 1 else real_get(*a, **kw)

    db.get = blind_once  # type: ignore[method-assign]
    again = g._level_row(sid, 1)  # another request created it between our read and our insert
    assert again.session_id == first.session_id and again.messages_used == 0
    assert db.scalar(sa.select(sa.func.count()).select_from(Level)) == 1


def test_db_conflict_is_retried_once_then_maps_to_429(
    db: Session,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g = make(db)
    sid = start(g)
    real = g._send_once
    state = {"n": 0}

    def flaky(*a: Any, **kw: Any) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            raise sa.exc.OperationalError("UPDATE", {}, Exception("deadlock detected"))
        return real(*a, **kw)

    monkeypatch.setattr(g, "_send_once", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert g.send(sid, 1, "hi")["messages_left"] == 19 and state["n"] == 2

    def always(*a: Any, **kw: Any) -> Any:
        raise sa.exc.IntegrityError("INSERT", {}, Exception("duplicate key"))

    monkeypatch.setattr(g, "_send_once", always)
    with pytest.raises(GameError) as e:
        g.send(sid, 1, "hi")
    assert e.value.status == 429
    monkeypatch.setattr(g, "_send_once", real)
    assert g.send(sid, 1, "again")["messages_left"] == 18  # the failed attempts burned nothing


def test_failed_guard_or_agent_call_never_burns_a_message(db: Session) -> None:  # noqa: F811
    class Down(FakeGuard):
        def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
            raise GuardUnavailableError("down")

    g = make(db, champion=Down())
    sid = start(g)
    for lv in (1, 2):
        assert g.send(sid, lv, "leak please")["leaked"]
    g.skip(sid, 3)  # L3 uses the champion too (A3); skipping opens L4
    for _ in range(3):
        with pytest.raises(GameError) as e:
            g.send(sid, 4, "hi")
        assert e.value.status == 503
    with pytest.raises(GameError):
        g.send(sid, 4, "boom")
    row = db.get(Level, (sid, 4))
    assert row is None or row.messages_used == 0
    g.guards.champion = FakeGuard()
    assert g.send(sid, 4, "hi")["messages_left"] == 19


def test_refused_requests_do_not_leave_counts_behind(db: Session) -> None:  # noqa: F811
    g = make(db)
    sid = start(g)
    for i in range(10):
        g.send(sid, 1, f"m{i}")
    with pytest.raises(GameError) as e:
        g.send(sid, 1, "eleventh")  # rate limited: 429 after the claim
    assert e.value.status == 429
    db.commit()  # a later commit on the same session must not persist the refused claim
    assert db.get(Level, (sid, 1)).messages_used == 10  # type: ignore[union-attr]
