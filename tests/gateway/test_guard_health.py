from __future__ import annotations

import threading
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Base
from breachlab.gateway.app import create_app
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guard_health import GuardProbe, placeholder_url
from breachlab.gateway.guards import RegexGuard
from breachlab.gateway.service import GuardSet
from breachlab.gateway.telemetry import Telemetry
from tests.gateway.test_service import FakeAgent

HEALTHY = {"ok": True, "models": {"champion": "v2", "challenger": None, "base": "base-0"}}


class Net:
    def __init__(self, *responses: Any) -> None:
        self.responses, self.calls = list(responses), []
        self.now = 0.0

    def get(self, url: str, **kw: Any) -> httpx.Response:
        self.calls.append((url, kw))
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[no-any-return]

    def clock(self) -> float:
        return self.now

    def probe(self, url: str = "https://guard.example", **kw: Any) -> GuardProbe:
        kw.setdefault("spawn", lambda fn: None)  # no background warm unless a test asks for one
        return GuardProbe(url, clock=self.clock, get=self.get, **kw)


def resp(status: int = 200, body: Any = HEALTHY) -> httpx.Response:
    return httpx.Response(status, json=body)


def test_healthy_guard_is_ok_and_probe_is_cached_for_the_ttl() -> None:
    net = Net(resp())
    p = net.probe()
    assert p.ok() is True and p.ok() is True and len(net.calls) == 1
    url, kw = net.calls[0]
    assert url == "https://guard.example/health" and kw["timeout"] <= 2.0
    net.now += 14
    assert p.ok() and len(net.calls) == 1
    net.now += 2
    assert p.ok() and len(net.calls) == 2


@pytest.mark.parametrize(
    "answer",
    [
        resp(503, {"detail": "waking up"}),
        resp(500),
        httpx.ConnectError("no route"),
        httpx.ReadTimeout("slow"),
        resp(200, {"ok": False}),
        resp(200, {"ok": True, "models": {"champion": None}}),
        resp(200, ["not", "a", "dict"]),
        httpx.Response(200, text="<html>Space is starting</html>"),
    ],
)
def test_unhealthy_answers_are_false(answer: Any) -> None:
    assert Net(answer).probe().ok() is False


def test_failure_is_cached_then_recovers_after_ttl() -> None:
    net = Net(resp(503), resp())
    p = net.probe()
    assert p.ok() is False and p.ok() is False and len(net.calls) == 1
    net.now += 3
    assert p.ok() is True


@pytest.mark.parametrize("url", ["", "https://guard.invalid", "https://x.y.invalid/", "not a url"])
def test_unset_or_placeholder_never_touches_the_network(url: str) -> None:
    net = Net(resp())
    assert net.probe(url).ok() is False and net.calls == []
    assert placeholder_url(url)
    assert not placeholder_url("https://giri-harsh-breachlab-guard.hf.space")


def test_token_is_sent_as_bearer_only_when_set() -> None:
    net = Net(resp())
    net.probe(token="s3cret").ok()
    assert net.calls[0][1]["headers"] == {"Authorization": "Bearer s3cret"}
    net2 = Net(resp())
    net2.probe().ok()
    assert net2.calls[0][1]["headers"] == {}


def test_probe_timeout_is_capped_at_two_seconds() -> None:
    net = Net(resp())
    net.probe(timeout=30).ok()
    assert net.calls[0][1]["timeout"] == 2.0


def test_concurrent_callers_do_not_wait_or_probe_twice() -> None:
    gate, started = threading.Event(), threading.Event()
    calls: list[int] = []

    def slow(url: str, **kw: Any) -> httpx.Response:
        calls.append(1)
        started.set()
        gate.wait(2)
        return resp()

    p = GuardProbe("https://guard.example", get=slow, spawn=lambda fn: None)
    t = threading.Thread(target=p.ok)
    t.start()
    started.wait(2)
    assert p.ok() is False  # second caller returns the cached value at once instead of waiting
    gate.set()
    t.join(3)
    assert len(calls) == 1 and p.ok() is True


def make_app(probe: GuardProbe | None, guard_url: str = "") -> TestClient:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    st = GatewaySettings(NEON_DATABASE_URL="sqlite://", GUARD_URL=guard_url)  # type: ignore[call-arg]
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=GuardSet(regex=RegexGuard()),
        agent=FakeAgent(),
        telemetry=Telemetry(),
        guard_probe=probe,
    )
    return TestClient(app)


def test_api_health_reports_guard_ok_and_keeps_guard_field() -> None:
    net = Net(resp())
    body = make_app(net.probe(), "https://guard.example").get("/api/health").json()
    assert body["ok"] is True and body["guard"] is True and body["guard_ok"] is True
    down = make_app(Net(resp(503)).probe(), "https://guard.example").get("/api/health").json()
    assert down["guard"] is True and down["guard_ok"] is False  # configured but not healthy


def test_api_health_defaults_false_without_url_or_with_placeholder() -> None:
    assert make_app(None).get("/api/health").json()["guard_ok"] is False
    assert make_app(None).get("/api/health").json()["guard"] is False
    placeholder = make_app(None, "https://guard-placeholder.invalid").get("/api/health").json()
    assert placeholder["guard"] is True and placeholder["guard_ok"] is False


def test_message_path_never_probes() -> None:
    net = Net(resp())
    client = make_app(net.probe(), "https://guard.example")
    sid = client.post("/api/session", json={"nickname": "Probe Free"}).json()["session_id"]
    client.post("/api/message", json={"session_id": sid, "level": 1, "message": "hello"})
    assert net.calls == []


# ---- guard_state: ok / warming / down (a cold guard is never "down") ----


def test_state_ok_when_the_guard_answers_in_time() -> None:
    p = Net(resp()).probe()
    assert p.state() == "ok" and p.ok() is True


def test_state_is_warming_when_the_probe_times_out_and_a_warm_is_fired_without_blocking() -> None:
    net = Net(httpx.ReadTimeout("cold"))
    fired: list[Any] = []
    p = net.probe(spawn=fired.append)
    assert p.state() == "warming" and p.ok() is False
    assert len(fired) == 1 and len(net.calls) == 1  # the warm is only scheduled, never awaited
    net.now += 3
    assert p.state() == "warming" and len(fired) == 1  # one warm in flight at a time


def test_the_warm_uses_the_long_timeout_and_success_turns_the_state_ok() -> None:
    net = Net(httpx.ReadTimeout("cold"), resp())
    p = net.probe(spawn=lambda fn: fn())
    # probe (2 s) times out, the synchronous warm (20 s) then succeeds; the next evaluation sees ok
    assert p.state() == "warming"
    assert net.calls[0][1]["timeout"] == 2.0 and net.calls[1][1]["timeout"] == 20.0
    net.now += 3
    assert p.state() == "ok"


def test_down_needs_a_stale_last_contact_and_a_failed_warm() -> None:
    net = Net(httpx.ReadTimeout("cold"))
    p = net.probe(spawn=lambda fn: fn())
    # never contacted + warm failed (both calls time out) -> down on the next evaluation
    assert p.state() == "warming" or p.state() == "down"
    net.now += 3
    assert p.state() == "down"
    assert make_app(p, "https://guard.example").get("/api/health").json()["guard_state"] in {
        "down",
        "warming",
    }


def test_recent_contact_keeps_it_warming_even_after_a_failed_warm() -> None:
    net = Net(resp(), httpx.ReadTimeout("slow"))
    p = net.probe(spawn=lambda fn: fn())
    assert p.state() == "ok"
    net.now += 100  # ok ttl expired, last contact 100 s ago (< 10 min)
    assert p.state() == "warming"
    net.now += 3
    assert p.state() == "warming"  # the warm failed but the last contact is recent
    net.now += 600  # last contact now older than 10 min and the warm failed
    assert p.state() == "down"
    net.now += 3
    assert p.state() == "down"


def test_down_clears_when_the_guard_comes_back() -> None:
    net = Net(httpx.ReadTimeout("x"), httpx.ReadTimeout("x"), httpx.ReadTimeout("x"), resp())
    p = net.probe(spawn=lambda fn: fn())
    p.state()
    net.now += 3
    assert p.state() == "down"
    net.now += 3
    assert p.state() == "ok"


def test_unconfigured_is_down_without_network() -> None:
    net = Net(resp())
    assert net.probe("").state() == "down" and net.calls == []


def test_evidence_from_the_message_path_and_warm_endpoint_counts() -> None:
    class G:
        warm_failed = True

        def __init__(self, age: float | None) -> None:
            self.age = age

        def last_ok_age(self) -> float | None:
            return self.age

    net = Net(httpx.ReadTimeout("x"))
    p = net.probe()
    p.attach(G(None))
    assert p.state() == "down"  # never answered anywhere + a failed /api/warm
    net2 = Net(httpx.ReadTimeout("x"))
    p2 = net2.probe()
    p2.attach(G(30.0))
    assert p2.state() == "warming"  # a message was answered 30 s ago: just cold


def test_api_health_reports_the_three_states_and_keeps_guard_ok() -> None:
    ok = make_app(Net(resp()).probe(), "https://guard.example").get("/api/health").json()
    assert ok["guard_state"] == "ok" and ok["guard_ok"] is True
    cold = (
        make_app(Net(httpx.ReadTimeout("c")).probe(), "https://guard.example")
        .get("/api/health")
        .json()
    )
    assert cold["guard_state"] == "warming" and cold["guard_ok"] is False and cold["ok"] is True
    off = make_app(None).get("/api/health").json()
    assert off["guard_state"] == "down" and off["guard_ok"] is False
