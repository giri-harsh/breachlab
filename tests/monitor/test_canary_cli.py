"""Canary CLI + OTLP pusher tests: no network (fake HTTP, in-process gateway)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest

from breachlab.gateway.guards import RegexGuard
from breachlab.monitor import canary_cli
from breachlab.monitor.canary import ATTACKS
from breachlab.monitor.canary_cli import Health, Report, check_health, derive_key, main, run
from breachlab.monitor.canary_otlp import (
    CANARY_METRICS,
    OtlpPusher,
    PushResult,
    basic_auth,
    pusher_from_env,
)
from tests.gateway.test_monitor import KeywordGuard, make_client


def metric_names(body: bytes) -> dict[str, float]:
    req = ExportMetricsServiceRequest.FromString(body)
    out: dict[str, float] = {}
    for rm in req.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                pts = m.gauge.data_points or m.sum.data_points
                p = pts[0]
                out[m.name] = p.as_double if p.HasField("as_double") else float(p.as_int)
    return out


class FakeOtlp:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [200])
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> tuple[int, str]:
        self.calls.append((url, body, headers))
        return (self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]), "resp"


def make_pusher(statuses: list[int] | None = None) -> tuple[OtlpPusher, FakeOtlp, list[float]]:
    fake, sleeps = FakeOtlp(statuses), []
    p = OtlpPusher(
        "https://otlp.example/otlp/", "123", "tok", post=fake, sleep=sleeps.append, attempts=3
    )
    return p, fake, sleeps


def getter(table: dict[str, list[tuple[int, dict[str, Any]]]]) -> canary_cli.Get:
    seen: dict[str, int] = {}

    def get(url: str, timeout: float) -> tuple[int, float, dict[str, Any]]:
        i = seen.get(url, 0)
        seen[url] = i + 1
        if url not in table:  # e.g. the guard's optional /vercel-info
            return 404, 0.1, {}
        rows = table[url]
        status, body = rows[min(i, len(rows) - 1)]
        return status, 1.5 if i == 0 else 0.25, body

    return get


OK = (200, {"ok": True})


def fake_post(guard_blind: bool = False, guard: str = "up", busy: int = 0) -> canary_cli.Post:
    """Gateway stand-in. L1 blocks nothing, L2 is the real regex, L4 blocks exactly the canary attacks
    (nothing when blind, a 503 "guard waking up" when ``guard == "offline"``). ``busy`` first agent calls 503."""
    state = {"busy": busy}

    def post(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
        assert headers["X-Canary"].count(".") == 1
        if url.endswith("/api/session"):
            return 200, {"session_id": "s1"}
        level, msg = body["level"], body["message"]
        if level == 4 and guard == "offline":
            return 503, {"detail": "guard waking up, retry in a few seconds"}
        if level == 4:
            blocked = msg in ATTACKS and not guard_blind
        elif level == 2:
            blocked = RegexGuard().classify(msg)[0] == "UNSAFE"
        else:
            blocked = False
        if not blocked and state["busy"] > 0:
            state["busy"] -= 1
            return 503, {"detail": "the vault is busy, try again"}
        return 200, {"blocked": blocked, "leaked": False}

    return post


def test_derive_key_matches_gateway_rule() -> None:
    assert derive_key({"GATEWAY_SECRET_KEY": "k", "ADMIN_PASSWORD": "pw"}) == "k"
    _, st = make_client(KeywordGuard())
    assert derive_key({"ADMIN_PASSWORD": "pw"}) == st.secret_key
    assert derive_key({}) == ""


def test_check_health_variants() -> None:
    sleeps: list[float] = []
    h = check_health("guard", "u", getter({"u": [OK]}), sleeps.append)
    assert h.ok and h.cold_s == 1.5 and h.warm_s == 0.25 and not sleeps
    h = check_health("guard", "u", getter({"u": [(503, {}), OK]}), sleeps.append)
    assert h.ok and sleeps == [5.0] and h.cold_s == pytest.approx(1.75)
    h = check_health("guard", "u", getter({"u": [(503, {})]}), sleeps.append)
    assert not h.ok and "503" in h.detail and h.warm_s is None
    h = check_health("guard", "u", getter({"u": [OK, (500, {})]}), sleeps.append)
    assert not h.ok and "warm" in h.detail
    h = check_health(
        "guard", "u", getter({"u": [(200, {"ok": False}), (200, {"ok": False})]}), sleeps.append
    )
    assert not h.ok


def env_for(**kw: str) -> dict[str, str]:
    return {"BASE_URL": "http://t", "ADMIN_PASSWORD": "pw", "GUARD_URL": "http://g", **kw}


def test_full_run_success_pushes_ok_metrics() -> None:
    pusher, fake, _ = make_pusher()
    rep = run(
        env_for(),
        get=getter({"http://g/health": [OK], "http://t/api/health": [OK]}),
        post=fake_post(),
        pusher=pusher,
        now=lambda: 1_700_000_000.0,
        probe_sleep=lambda _: None,
    )
    assert rep.ok and len(rep.probes) == 20 and rep.push == PushResult(True, 200, "accepted")
    url, body, headers = fake.calls[0]
    assert url == "https://otlp.example/otlp/v1/metrics"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"123:tok").decode()
    got = metric_names(body)
    assert got["canary_ok_total"] == 1 and got["canary_failures_total"] == 0
    assert got["canary_last_success_timestamp"] == 1_700_000_000.0
    assert got["guard_cold_start_seconds"] == 1.5 and got["guard_warm_latency_seconds"] == 0.25
    assert got["guard_unavailable_total"] == 0 and got["gateway_cold_start_seconds"] == 1.5
    assert set(got) <= set(CANARY_METRICS)


def test_guard_down_and_blind_guard_fail_and_push_failures() -> None:
    pusher, fake, _ = make_pusher()
    rep = run(
        env_for(),
        get=getter({"http://g/health": [(503, {})], "http://t/api/health": [OK]}),
        post=fake_post(guard_blind=True),
        pusher=pusher,
        sleep=lambda _: None,
        probe_sleep=lambda _: None,
    )
    assert not rep.ok
    assert rep.failures[0].startswith("guard health") and len(rep.failures) == 11
    got = metric_names(fake.calls[0][1])
    assert got["canary_failures_total"] == 11 and got["canary_ok_total"] == 0
    assert got["guard_unavailable_total"] == 1 and "canary_last_success_timestamp" not in got
    assert "guard_cold_start_seconds" not in got  # only recorded for healthy checks


def test_gateway_down_skips_probes_and_no_guard_url() -> None:
    called: list[str] = []

    def post(url: str, body: dict[str, Any], hdr: dict[str, str]) -> tuple[int, dict[str, Any]]:
        called.append(url)
        return 200, {}

    rep = run(
        {"BASE_URL": "http://t", "ADMIN_PASSWORD": "pw"},
        get=getter({"http://t/api/health": [(500, {})]}),
        post=post,
        sleep=lambda _: None,
    )
    assert not called and rep.probes == [] and [h.name for h in rep.health] == ["gateway"]
    assert rep.failures == ["gateway health: http 500, then 500"] and rep.push is None


def test_default_base_url_and_level(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def get(url: str, timeout: float) -> tuple[int, float, dict[str, Any]]:
        seen.append(url)
        return 500, 0.1, {}

    run({"BASE_URL": "", "CANARY_LEVEL": ""}, get=get, sleep=lambda _: None)
    assert seen[0] == canary_cli.DEFAULT_BASE_URL + "/api/health"


def test_main_not_configured_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([], env={}) == 2
    assert "required" in capsys.readouterr().out
    summary = tmp_path / "s.md"
    pusher, _, _ = make_pusher([500])
    kw: dict[str, Any] = {
        "get": getter({"http://g/health": [OK], "http://t/api/health": [OK]}),
        "post": fake_post(),
        "pusher": pusher,
        "probe_sleep": lambda _: None,
        "sleep": lambda _: None,
    }
    code = main(["--json"], env=env_for(GITHUB_STEP_SUMMARY=str(summary)), **kw)
    out = capsys.readouterr().out
    assert code == 0 and '"ok": true' in out and "::warning::OTLP push failed" in out
    assert "PASS" in summary.read_text() and "passed: 20/20" in summary.read_text()


def test_main_failure_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    kw: dict[str, Any] = {
        "get": getter({"http://t/api/health": [(500, {})]}),
        "sleep": lambda _: None,
    }
    code = main([], env={"BASE_URL": "http://t", "GATEWAY_SECRET_KEY": "k"}, **kw)
    out = capsys.readouterr().out
    assert code == 1 and "FAIL" in out and "skipped (no credentials)" in out


def test_ingest_test_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--ingest-test"], env={}) == 2
    pusher, fake, _ = make_pusher()
    assert main(["--ingest-test"], env={}, pusher=pusher, now=lambda: 42.0) == 0
    assert metric_names(fake.calls[0][1]) == {"breachlab_otlp_ingest_test": 42.0}
    assert "http=200" in capsys.readouterr().out
    bad, _, _ = make_pusher([401])
    assert main(["--ingest-test"], env={}, pusher=bad) == 1


def test_summary_md_without_health() -> None:
    md = canary_cli.summary_md(Report(health=[Health("guard", False, None, None, "down")]))
    assert "| guard | down | - | - |" in md and "PASS" in md


def test_pusher_retries_and_gives_up() -> None:
    p, fake, sleeps = make_pusher([429, 503, 200])
    p.gauge("x", 1.0)
    assert p.flush().ok and sleeps == [2.0, 4.0] and len(fake.calls) == 3
    p, fake, sleeps = make_pusher([500])
    p.counter("x")
    res = p.flush()
    assert not res.ok and res.status == 500 and len(fake.calls) == 3 and sleeps == [2.0, 4.0]
    p, fake, sleeps = make_pusher([401])
    p.gauge("x", 1.0)
    assert p.flush() == PushResult(False, 401, "resp") and len(fake.calls) == 1 and not sleeps


def test_pusher_nothing_recorded_and_env() -> None:
    p, fake, _ = make_pusher()
    assert p.flush() == PushResult(False, None, "nothing recorded") and not fake.calls
    assert pusher_from_env({"GRAFANA_OTLP_ENDPOINT": "e"}) is None
    q = pusher_from_env(
        {
            "GRAFANA_OTLP_ENDPOINT": "https://e/otlp",
            "GRAFANA_OTLP_INSTANCE_ID": "1",
            "GRAFANA_OTLP_TOKEN": "t",
        }
    )
    assert q is not None and q.url == "https://e/otlp/v1/metrics"
    assert basic_auth("a", "b") == "Basic YTpi"
