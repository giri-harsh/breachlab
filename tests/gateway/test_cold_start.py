"""Cold start handling (ADR-0085): cold timeout, 202 warming within the grace window, 503 after it, /api/warm."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Base
from breachlab.gateway.app import create_app
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guards import GuardUnavailableError, HttpGuard, RegexGuard
from breachlab.gateway.service import GameError, GuardSet
from breachlab.gateway.telemetry import Telemetry
from tests.gateway.test_service import FakeAgent, FakeGuard, clear, db, make, start  # noqa: F401

OK_BODY = {
    "label": "SAFE",
    "category": "none",
    "p_unsafe": 0.01,
    "confidence": 0.01,
    "version": "v",
}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class FakeClient:
    """An httpx.Client stand-in that records the timeout of every call and can be switched between up and down."""

    def __init__(self) -> None:
        self.up = True
        self.timeouts: list[float] = []
        self.gets = 0

    def post(self, url: str, **kw: Any) -> httpx.Response:
        self.timeouts.append(kw["timeout"])
        if not self.up:
            raise httpx.ReadTimeout("booting")
        return httpx.Response(200, json=OK_BODY, request=httpx.Request("POST", url))

    def get(self, url: str, **kw: Any) -> httpx.Response:
        self.gets += 1
        if not self.up:
            raise httpx.ReadTimeout("booting")
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))


def guard(clock: Clock, client: FakeClient, **kw: Any) -> HttpGuard:
    return HttpGuard("http://g.invalid", 8.0, retries=1, client=client, clock=clock, **kw)  # type: ignore[arg-type]


def test_first_call_is_cold_then_steady_then_cold_again_after_idle() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    assert g.is_cold()
    g.score("hi")
    clock.t += 30
    g.score("hi")
    clock.t += 300  # idle longer than cold_after_s (120)
    assert g.is_cold()
    g.score("hi")
    assert client.timeouts == [20.0, 8.0, 20.0]


def test_cold_failure_is_warming_inside_the_grace_window_and_down_after_it() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    client.up = False
    with pytest.raises(GuardUnavailableError) as first:
        g.score("hi")
    assert first.value.warming is True
    assert client.timeouts == [20.0, 20.0]  # cold timeout and exactly one retry
    clock.t += 30
    with pytest.raises(GuardUnavailableError) as second:
        g.score("hi")
    assert second.value.warming is True
    clock.t += 40  # 70 s after the first failure: past the 60 s grace window
    with pytest.raises(GuardUnavailableError) as third:
        g.score("hi")
    assert third.value.warming is False  # truly down: fail closed (503)


def test_failure_of_a_warm_guard_is_not_warming() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    g.score("hi")
    client.up = False
    clock.t += 10
    with pytest.raises(GuardUnavailableError) as e:
        g.score("hi")
    assert e.value.warming is False
    assert client.timeouts[-2:] == [8.0, 8.0]  # steady state keeps the 8 s timeout


def test_success_resets_the_grace_window() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    client.up = False
    with pytest.raises(GuardUnavailableError):
        g.score("hi")
    client.up = True
    g.score("hi")
    clock.t += 500
    client.up = False
    with pytest.raises(GuardUnavailableError) as e:  # cold again, a fresh window
        g.score("hi")
    assert e.value.warming is True


def test_warm_pings_health_and_is_cached() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    assert g.warm()["status"] == "warm"
    assert g.warm()["status"] == "warm"
    assert client.gets == 1  # a flood of page loads costs one ping
    assert not g.is_cold()  # the answer counts as a sign of life
    clock.t += 21
    g.warm()
    assert client.gets == 2


def test_warm_reports_warming_then_down() -> None:
    clock, client = Clock(), FakeClient()
    client.up = False
    g = guard(clock, client)
    assert g.warm()["status"] == "warming"
    clock.t += 100
    assert g.warm()["status"] == "down"


class WarmingGuard(FakeGuard):
    def score(self, text: str, kind: str = "input", alias: str = "champion") -> Any:
        raise GuardUnavailableError("timeout", warming=True)


def test_service_answers_202_and_burns_no_message_while_warming(db: Session) -> None:  # noqa: F811
    class Metrics:
        def __init__(self) -> None:
            self.warming: list[tuple[int, str, str]] = []
            self.unavailable: list[tuple[int, str, str]] = []

        def guard_unavailable(self, level: int, alias: str, kind: str) -> None:
            self.unavailable.append((level, alias, kind))

        def guard_warming(self, level: int, alias: str, kind: str) -> None:
            self.warming.append((level, alias, kind))

    g = make(db, champion=WarmingGuard())
    tel = Telemetry(Metrics())
    g.telemetry = tel
    sid = start(g)
    for lv in (1, 2):
        clear(g, sid, lv)
    g.skip(sid, 3)  # L3 shares the (warming) champion guard since A3; a skip opens L4
    used = g.state(sid)["messages_used"]
    with pytest.raises(GameError) as e:
        g.send(sid, 4, "hello there")
    assert e.value.status == 202 and "waking up" in e.value.detail
    assert (
        g.state(sid)["messages_used"] == used
    )  # the half-written attempt was rolled back: no message burned
    assert g.state(sid)["cleared"] == 2
    assert tel.metrics.warming == [(4, "champion", "input")] and tel.metrics.unavailable == []  # type: ignore[union-attr]


def make_app(guards: GuardSet) -> TestClient:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    st = GatewaySettings(ADMIN_PASSWORD="pw", NEON_DATABASE_URL="sqlite://")  # type: ignore[call-arg]
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=guards,
        agent=FakeAgent(),
        telemetry=Telemetry(),
    )
    return TestClient(app)


def test_message_endpoint_returns_202_json_with_retry_after() -> None:
    c = make_app(GuardSet(regex=RegexGuard(), base=WarmingGuard(), champion=WarmingGuard()))
    sid = c.post("/api/session", json={"nickname": "Cold Start"}).json()["session_id"]
    for lv in (1, 2):
        assert c.post(
            "/api/message", json={"session_id": sid, "level": lv, "message": "leak"}
        ).json()["leaked"]
    r = c.post("/api/message", json={"session_id": sid, "level": 3, "message": "hello"})
    assert r.status_code == 202
    assert r.json()["status"] == "warming" and r.json()["retry_after_s"] == 5
    assert r.headers["retry-after"] == "5"


def test_api_warm_endpoint_states_and_no_hosts_leaked() -> None:
    clock, client = Clock(), FakeClient()
    g = guard(clock, client)
    c = make_app(GuardSet(regex=RegexGuard(), base=g, champion=g))
    r = c.get("/api/warm")
    assert r.status_code == 200 and r.json()["status"] == "warm"
    assert list(r.json()["guards"]) == ["champion"]  # one distinct host, deduplicated
    assert "g.invalid" not in r.text
    client.up = False
    clock.t += 500
    r2 = c.get("/api/warm")
    assert r2.status_code == 202 and r2.json()["status"] == "warming"


def test_api_warm_without_a_guard_says_none() -> None:
    c = make_app(GuardSet(regex=RegexGuard()))
    r = c.get("/api/warm")
    assert r.status_code == 200 and r.json()["status"] == "none"


def test_separate_base_host_is_warmed_too() -> None:
    clock = Clock()
    c1, c2 = FakeClient(), FakeClient()
    champion, base = guard(clock, c1), HttpGuard("http://b.invalid", 8.0, client=c2, clock=clock)  # type: ignore[arg-type]
    c = make_app(GuardSet(regex=RegexGuard(), base=base, champion=champion))
    r = c.get("/api/warm").json()
    assert set(r["guards"]) == {"champion", "base"} and c1.gets == 1 and c2.gets == 1
