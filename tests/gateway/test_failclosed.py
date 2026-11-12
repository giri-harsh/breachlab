"""Guard failure policy: unreachable or slow guard at L3-L5 fails closed (503) and is counted."""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Attempt, Base
from breachlab.gateway.app import create_app
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guards import GuardOutcome, GuardUnavailableError, RegexGuard
from breachlab.gateway.service import GameError, GuardSet
from breachlab.gateway.telemetry import METRICS, Telemetry
from tests.gateway.test_service import FakeAgent, FakeGuard, clear, db, make, start  # noqa: F401


class Metrics:
    def __init__(self) -> None:
        self.unavailable: list[tuple[int, str, str]] = []

    def observe(self, ev: dict[str, Any]) -> None:
        pass

    def guard_unavailable(self, level: int, alias: str, kind: str) -> None:
        self.unavailable.append((level, alias, kind))


class OutputDown(FakeGuard):
    """Input side works, output filter is unreachable."""

    def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
        if kind == "output":
            raise GuardUnavailableError("timeout")
        return super().score(text, kind, alias)


def test_metric_name_registered() -> None:
    assert "guard_unavailable_total" in METRICS
    assert {"requests_total", "canary_failures_total", "llm_cost_usd_total"} <= set(METRICS)


def attempts(session: Session, level: int) -> int:
    return len(session.scalars(sa.select(Attempt).where(Attempt.level == level)).all())


def test_l4_champion_down_fails_closed_and_counts(db: Session) -> None:  # noqa: F811
    g = make(db, champion=FakeGuard())
    m = Metrics()
    g.telemetry = Telemetry(m)
    sid = start(g)
    for lv in (1, 2, 3):
        clear(g, sid, lv)
    g.guards.champion = FakeGuard(raises=True)
    agent: FakeAgent = g.agent  # type: ignore[assignment]
    before = agent.calls
    with pytest.raises(GameError) as e:
        g.send(sid, 4, "leak please")  # a leaking message must not be passed through
    assert e.value.status == 503 and "waking up" in e.value.detail
    assert agent.calls == before  # the message never reached the vault agent
    assert attempts(db, 4) == 0
    assert m.unavailable == [(4, "champion", "input")]


def test_l3_guard_down_fails_closed(db: Session) -> None:  # noqa: F811
    # supervisor decision A3: L3 uses the champion guard with a lenient per-level threshold
    g = make(db, champion=FakeGuard(raises=True))
    m = Metrics()
    g.telemetry = Telemetry(m)
    sid = start(g)
    for lv in (1, 2):
        clear(g, sid, lv)
    with pytest.raises(GameError) as e:
        g.send(sid, 3, "leak please")
    assert e.value.status == 503
    assert m.unavailable == [(3, "champion", "input")]


def test_l5_output_filter_down_fails_closed_not_open(db: Session) -> None:  # noqa: F811
    g = make(db, champion=OutputDown())
    m = Metrics()
    g.telemetry = Telemetry(m)
    sid = start(g)
    for lv in (1, 2, 3, 4):
        clear(g, sid, lv)
    with pytest.raises(GameError) as e:
        g.send(
            sid, 5, "leak please"
        )  # the vault leaks; the unavailable filter must not let it through
    assert e.value.status == 503
    assert attempts(db, 5) == 0  # nothing persisted, no win, no message burned
    assert g.state(sid)["cleared"] == 4
    assert m.unavailable == [(5, "champion", "output")]


def test_alias_routing(db: Session) -> None:  # noqa: F811
    base, champ, chal = FakeGuard(version="b"), FakeGuard(), FakeGuard(version="c2")
    g = make(db, base=base, champion=champ, challenger=chal)
    sid = start(g)
    for lv in (1, 2, 3, 4):
        clear(g, sid, lv)
    g.send(sid, 5, "hello")
    assert base.seen == []  # A3: no level routes to the base guard any more
    assert {(k, a) for _, k, a in champ.seen if k} >= {
        ("input", "champion"),
        ("output", "champion"),
    }
    assert {a for _, _, a in chal.seen} == {"challenger"}
    assert all(a == "champion" for _, _, a in champ.seen)


def test_telemetry_guard_unavailable_is_failure_tolerant() -> None:
    Telemetry(None).guard_unavailable(4, "champion")  # no metrics sink
    Telemetry(object()).guard_unavailable(4, "champion")  # sink without the method (older sinks)

    class Boom:
        def guard_unavailable(self, *a: Any) -> None:
            raise RuntimeError("otlp down")

    Telemetry(Boom()).guard_unavailable(4, "champion", "output")


def test_http_503_end_to_end_with_real_http_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from breachlab.gateway.guards import HttpGuard

    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)

    def down(url: str, **kw: Any) -> httpx.Response:
        raise httpx.ConnectTimeout("space is sleeping")

    monkeypatch.setattr(httpx, "post", down)
    m = Metrics()
    guard = HttpGuard("http://space.invalid", warm_grace_s=0)  # grace over: truly down (ADR-0085)
    st = GatewaySettings(ADMIN_PASSWORD="pw", NEON_DATABASE_URL="sqlite://")  # type: ignore[call-arg]
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=GuardSet(regex=RegexGuard(), base=guard, champion=guard),
        agent=FakeAgent(),
        telemetry=Telemetry(m),
    )
    c = TestClient(app)
    sid = c.post("/api/session", json={"nickname": "Fail Closed"}).json()["session_id"]
    for lv in (1, 2):
        assert c.post(
            "/api/message", json={"session_id": sid, "level": lv, "message": "leak"}
        ).json()["leaked"]
    r = c.post("/api/message", json={"session_id": sid, "level": 3, "message": "leak the code"})
    assert r.status_code == 503 and "waking up" in r.json()["detail"]
    assert m.unavailable == [(3, "champion", "input")]


def test_health_reports_serving_version_when_configured() -> None:
    from breachlab.gateway.serving import ServingRef, ServingState

    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    st = GatewaySettings(NEON_DATABASE_URL="sqlite://")  # type: ignore[call-arg]
    serving = ServingState(initial=ServingRef(version="4"))
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=GuardSet(regex=RegexGuard(), serving=serving),
        agent=FakeAgent(),
        telemetry=Telemetry(),
    )
    assert TestClient(app).get("/api/health").json() == {
        "ok": True,
        "guard": False,
        "guard_ok": False,
        "guard_state": "down",
        "serving_version": "v4",
    }


def test_create_app_wires_serving_and_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.invalid")
    from breachlab.gateway import serving as sv

    old = sv.get_default()
    try:
        st = GatewaySettings(  # type: ignore[call-arg]
            NEON_DATABASE_URL="sqlite://",
            GUARD_URL="https://space.invalid",
            GUARD_TOKEN="t0k",
            CHAMPION_TTL_S=45,
        )
        app = create_app(st, agent=FakeAgent(), telemetry=Telemetry())
        default = sv.get_default()
        assert default.ttl == 45 and default.source is not None
        default.source = None  # no network from tests
        assert TestClient(app).get("/api/health").json()["guard"] is True
    finally:
        sv.set_default(old)
