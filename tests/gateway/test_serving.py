"""Champion hot-swap: poll-on-request TTL cache, swap/rollback, poll failure, zero request loss."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Attempt
from breachlab.gateway.guards import GuardOutcome
from breachlab.gateway.service import GameService, GuardSet
from breachlab.gateway.serving import (
    RegistryChampionSource,
    RestChampionSource,
    ServingRef,
    ServingState,
    get_default,
    resolve_label,
    serving_version,
    set_default,
    source_from_env,
)
from tests.gateway.test_service import FakeAgent, FakeGuard, db, start  # noqa: F401


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class Source:
    """Scripted champion source: ``version`` is what MLflow says; ``fail`` makes it unreachable."""

    def __init__(self, version: str = "1") -> None:
        self.version, self.fail, self.calls = version, False, 0

    def fetch(self) -> ServingRef:
        self.calls += 1
        if self.fail:
            raise ConnectionError("mlflow unreachable")
        return ServingRef(version=self.version, hf_repo=f"org/guard-v{self.version}", kernel="k")


def state(src: Source, clk: Clock, **kw: Any) -> ServingState:
    return ServingState(src, ttl=30, clock=clk, **kw)


def test_first_request_polls_then_ttl_caches() -> None:
    src, clk = Source("1"), Clock()
    st = state(src, clk)
    assert st.current().label == "unknown"  # nothing polled yet
    assert st.snapshot().label == "v1" and src.calls == 1
    clk.t += 29.9
    src.version = "2"
    assert st.snapshot().label == "v1" and src.calls == 1  # TTL not elapsed: no MLflow call
    clk.t += 0.2
    assert st.snapshot().label == "v2" and src.calls == 2
    assert st.current().hf_repo == "org/guard-v2" and st.current().kernel == "k"


def test_swap_rollback_and_observer() -> None:
    src, clk = Source("1"), Clock()
    swaps: list[tuple[str, str]] = []
    st = state(src, clk, on_swap=lambda a, b: swaps.append((a.label, b.label)))
    st.snapshot()
    for v in ("2", "1"):  # promote 2, then rollback to 1
        src.version = v
        clk.t += 31
        assert st.snapshot().version == v
    assert swaps == [("unknown", "v1"), ("v1", "v2"), ("v2", "v1")]
    clk.t += 31
    assert st.refresh(force=True) is False  # same version: no swap event
    assert len(swaps) == 3


def test_observer_failure_does_not_break_serving() -> None:
    def boom(a: ServingRef, b: ServingRef) -> None:
        raise RuntimeError("observer down")

    st = state(Source("3"), Clock(), on_swap=boom)
    assert st.snapshot().version == "3"


def test_poll_failure_keeps_old_champion_and_retries_after_ttl() -> None:
    src, clk = Source("1"), Clock()
    st = state(src, clk)
    st.snapshot()
    src.fail, src.version = True, "2"
    clk.t += 31
    assert st.snapshot().label == "v1"  # MLflow unreachable -> keep v1
    assert st.failures == 1 and st.last_error == "ConnectionError" and src.calls == 2
    assert (
        st.snapshot().label == "v1" and src.calls == 2
    )  # failed poll is not retried inside the TTL
    clk.t += 31
    assert st.snapshot().label == "v1" and st.failures == 2
    src.fail = False
    clk.t += 31
    assert st.snapshot().label == "v2" and st.failures == 0 and st.last_error is None


def test_failure_before_first_success_reports_unknown_and_falls_back() -> None:
    src = Source()
    src.fail = True
    st = state(src, Clock())
    assert st.snapshot().label == "unknown"
    assert serving_version(st, fallback="guard-reported") == "guard-reported"
    assert serving_version(st) == "unknown"


def test_no_source_is_inert() -> None:
    st = ServingState()
    assert st.snapshot().label == "unknown" and st.refresh() is False
    st.start_background()  # nothing to poll
    assert st._thread is None


def test_serving_version_single_source_of_truth() -> None:
    old = get_default()
    try:
        st = state(Source("7"), Clock())
        set_default(st)
        assert serving_version() == "v7" == serving_version(st) == resolve_label(st.current())
        assert (
            resolve_label(ServingRef(), "x") == "x"
            and resolve_label(ServingRef(version="2"), "x") == "v2"
        )
    finally:
        set_default(old)


def test_only_one_concurrent_refresher_and_refs_stay_consistent() -> None:
    gate, started = threading.Event(), threading.Event()
    in_flight = {"now": 0, "max": 0}
    lock = threading.Lock()

    class Slow:
        def fetch(self) -> ServingRef:
            with lock:
                in_flight["now"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["now"])
            started.set()
            gate.wait(2)
            with lock:
                in_flight["now"] -= 1
            return ServingRef(version="9")

    st = ServingState(Slow(), ttl=30)
    seen: list[str] = []
    ts = [threading.Thread(target=lambda: seen.append(st.snapshot().label)) for _ in range(8)]
    for t in ts:
        t.start()
    started.wait(2)
    time.sleep(0.05)
    gate.set()
    for t in ts:
        t.join(3)
    assert in_flight["max"] == 1  # the other 7 requests did not wait for, or repeat, the poll
    assert len(seen) == 8 and set(seen) <= {"unknown", "v9"}
    assert st.snapshot().label == "v9"


def test_background_variant_swaps() -> None:
    src = Source("1")
    st = ServingState(src, ttl=0.01)
    st.start_background(0.01)
    st.start_background(0.01)  # idempotent
    try:
        deadline = time.time() + 3
        while st.current().label != "v1" and time.time() < deadline:
            time.sleep(0.01)
        src.version = "2"
        while st.current().label != "v2" and time.time() < deadline:
            time.sleep(0.01)
        assert st.current().label == "v2"
    finally:
        st.stop_background()


def test_registry_source_reads_mlflow_client_tags() -> None:
    calls: list[tuple[str, str]] = []

    class C:
        def get_model_version_by_alias(self, name: str, alias: str) -> Any:
            calls.append((name, alias))
            return SimpleNamespace(
                version=4, tags={"hf_repo": "Pseudo-Harsh/g", "kernel": "giri/k"}
            )

    ref = RegistryChampionSource(C()).fetch()
    assert calls == [("breachlab-guard", "champion")]
    assert (ref.version, ref.hf_repo, ref.kernel) == ("4", "Pseudo-Harsh/g", "giri/k")


def test_rest_source_parses_mlflow_rest_and_sends_auth() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["params"] = (
            req.url.path,
            req.headers.get("authorization"),
            dict(req.url.params),
        )
        return httpx.Response(
            200,
            json={
                "model_version": {
                    "version": "5",
                    "tags": [{"key": "hf_repo", "value": "o/g"}, {"key": "kernel", "value": "k5"}],
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    src = RestChampionSource(
        "https://dagshub.example/x.mlflow/", username="u", password="p", client=client
    )
    ref = src.fetch()
    assert (ref.version, ref.hf_repo, ref.kernel) == ("5", "o/g", "k5")
    assert seen["url"].endswith("/api/2.0/mlflow/registered-models/alias")
    assert seen["params"] == {"name": "breachlab-guard", "alias": "champion"}
    assert seen["auth"].startswith("Basic ")

    bearer = RestChampionSource("https://m", token="tok", client=client)
    bearer.fetch()
    assert seen["auth"] == "Bearer tok"

    def not_found(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})

    gone = RestChampionSource(
        "https://m", client=httpx.Client(transport=httpx.MockTransport(not_found))
    )
    with pytest.raises(httpx.HTTPStatusError):
        gone.fetch()


def test_rest_source_default_client_uses_httpx_get(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model_version": {"version": "8", "tags": {"hf_repo": "r"}}},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    ref = RestChampionSource("https://m").fetch()
    assert ref.version == "8" and ref.hf_repo == "r"


def test_source_from_env() -> None:
    assert source_from_env({}) is None
    s = source_from_env(
        {
            "MLFLOW_TRACKING_URI": "https://m/",
            "MLFLOW_TRACKING_USERNAME": "u",
            "MLFLOW_TRACKING_PASSWORD": "p",
        }
    )
    assert isinstance(s, RestChampionSource) and s.base == "https://m" and s.auth == ("u", "p")


# ---- through the game pipeline


def serving_service(session: Session, st: ServingState, champion: FakeGuard) -> GameService:
    gs = GuardSet(
        regex=FakeGuard(), base=FakeGuard(version="base-0"), champion=champion, serving=st
    )
    return GameService(session, gs, FakeAgent(), "k" * 16, "salt")


def clear_to_l4(g: GameService, sid: str) -> None:
    for lv in (1, 2, 3):
        assert g.send(sid, lv, "leak please")["leaked"]


def test_pipeline_reports_serving_version_and_follows_swap_and_rollback(db: Session) -> None:  # noqa: F811
    src, clk = Source("1"), Clock()
    st = state(src, clk)
    g = serving_service(db, st, FakeGuard(version="space-says-x"))
    sid = start(g)
    clear_to_l4(g, sid)
    r = g.send(sid, 4, "hello")
    assert r["guard_version"] == "v1"  # registry champion, not the Space's own string
    src.version, clk.t = "2", clk.t + 31  # promote
    assert g.send(sid, 4, "hello again")["guard_version"] == "v2"
    src.version, clk.t = "1", clk.t + 31  # rollback
    assert g.send(sid, 4, "hello thrice")["guard_version"] == "v1"
    versions = [
        v for (v,) in db.execute(sa.select(Attempt.guard_version).where(Attempt.level == 4))
    ]
    assert versions == ["v1", "v2", "v1"]
    assert (
        db.scalar(sa.select(Attempt.guard_version).where(Attempt.level == 3)) == "v1"
    )  # A3: L3 is the champion with a lenient threshold, so it carries the serving version too


def test_registry_outage_mid_game_keeps_champion_and_requests_succeed(db: Session) -> None:  # noqa: F811
    src, clk = Source("3"), Clock()
    st = state(src, clk)
    g = serving_service(db, st, FakeGuard())
    sid = start(g)
    clear_to_l4(g, sid)
    assert g.send(sid, 4, "hi")["guard_version"] == "v3"
    src.fail, clk.t = True, clk.t + 31
    assert g.send(sid, 4, "hi again")["guard_version"] == "v3"  # no request lost, still v3


def test_swap_during_request_does_not_lose_or_mix_it(db: Session) -> None:  # noqa: F811
    src, clk = Source("1"), Clock()
    st = state(src, clk)

    class SwapsMidFlight(FakeGuard):
        armed = False

        def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
            if self.armed and kind == "input":
                self.armed = False
                src.version = "2"
                assert st.refresh(force=True)  # the swap lands while this request is in flight
            return super().score(text, kind, alias)

    champ = SwapsMidFlight()
    g = serving_service(db, st, champ)
    sid = start(g)
    clear_to_l4(g, sid)
    champ.armed = True
    r = g.send(sid, 4, "hello")
    assert r["guard_version"] == "v1" and not r["blocked"]  # finished on the old reference
    assert g.send(sid, 4, "next")["guard_version"] == "v2"  # the next request is on the new one


def test_leaderboard_tags_serving_version(db: Session) -> None:  # noqa: F811
    st = state(Source("6"), Clock())
    g = serving_service(db, st, FakeGuard())
    sid = start(g)
    clear_to_l4(g, sid)
    g.send(sid, 4, "leak please")
    assert g.leaderboard()[0]["guard_version"] == "v6"
    assert g.hall_of_breaches()[0]["guard_version"] == "v6"
