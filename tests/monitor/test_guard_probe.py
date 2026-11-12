"""Direct guard probe (lane E3, ADR-0082): cold-start timeout is a warning, two failures are a failure. No network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from breachlab.monitor import guard_probe
from breachlab.monitor.canary_cli import Report, run, summary_md
from breachlab.monitor.guard_probe import GuardGet, GuardReply, adapt_get, probe_guard_host
from tests.monitor.test_canary_cli import fake_post, getter, make_pusher, metric_names

NOSLEEP: dict[str, Any] = {"sleep": lambda _: None, "probe_sleep": lambda _: None}
G = "http://g"
GATEWAY_OK = {"http://t/api/health": [(200, {"ok": True, "guard": True})]}
HEALTH_OK = GuardReply(200, 0.3, {"ok": True}, {"x-guard-cold": "0", "x-guard-boot-s": "9.701"})
COLD_OK = GuardReply(
    200,
    12.8,
    {"ok": True},
    {"x-guard-cold": "1", "x-guard-boot-s": "9.701", "x-guard-wall-ms": "1"},
)
TIMEOUT = GuardReply(599, 25.0, timed_out=True)
INFO = GuardReply(
    200,
    0.2,
    {"region": "iad1", "instance_age_s": 14.5, "served": 2, "boot_error": None, "fetch": {}},
)


def script(table: Mapping[str, list[GuardReply]]) -> tuple[GuardGet, list[tuple[str, float]]]:
    calls: list[tuple[str, float]] = []
    seen: dict[str, int] = {}

    def get(url: str, timeout: float) -> GuardReply:
        calls.append((url, timeout))
        i = seen.get(url, 0)
        seen[url] = i + 1
        rows = table[url]
        return rows[min(i, len(rows) - 1)]

    return get, calls


def env(**kw: str) -> dict[str, str]:
    return {"BASE_URL": "http://t", "ADMIN_PASSWORD": "pw", **kw}


def test_warm_guard_reports_ok_info_and_no_warning() -> None:
    get, calls = script({G + "/health": [HEALTH_OK], G + "/vercel-info": [INFO]})
    p = probe_guard_host(G, get, lambda _: None)
    assert p.ok and p.detail == "ok" and not p.warnings
    assert p.cold is False and p.boot_s == 9.701  # a warm instance still reports its own boot time
    assert p.region == "iad1" and p.served == 2 and p.instance_age_s == 14.5 and p.info_ok
    assert [u for u, _ in calls] == [G + "/health", G + "/health", G + "/vercel-info"]


def test_cold_start_within_the_timeout_is_ok_and_reports_boot_seconds_from_the_header() -> None:
    get, _ = script({G + "/health": [COLD_OK, HEALTH_OK], G + "/vercel-info": [INFO]})
    p = probe_guard_host(G, get, lambda _: None)
    assert p.ok and p.cold is True and p.boot_s == 9.701 and p.first_s == 12.8
    assert p.warm_s == 0.3 and not p.warnings


def test_boot_seconds_prefer_vercel_info_when_it_has_them() -> None:
    info = GuardReply(200, 0.1, {"boot_s": 7.5, "served": 1})
    get, _ = script({G + "/health": [COLD_OK, HEALTH_OK], G + "/vercel-info": [info]})
    assert probe_guard_host(G, get, lambda _: None).boot_s == 7.5


def test_single_cold_start_timeout_is_a_warning_not_a_failure() -> None:
    sleeps: list[float] = []
    get, calls = script({G + "/health": [TIMEOUT, COLD_OK, HEALTH_OK], G + "/vercel-info": [INFO]})
    p = probe_guard_host(G, get, sleeps.append)
    assert p.ok and p.cold is True
    assert len(p.warnings) == 1 and "cold-start timeout" in p.warnings[0]
    assert sleeps == [guard_probe.RETRY_GAP_S]
    assert p.first_s == 25.0 + 12.8
    assert calls[0][1] == guard_probe.FIRST_TIMEOUT_S


def test_non_timeout_blip_then_ok_is_also_only_a_warning() -> None:
    get, _ = script({G + "/health": [GuardReply(502, 0.4), HEALTH_OK], G + "/vercel-info": [INFO]})
    p = probe_guard_host(G, get, lambda _: None)
    assert p.ok and "failed once (http 502)" in p.warnings[0]


def test_two_failures_in_a_row_are_a_failure() -> None:
    get, _ = script({G + "/health": [TIMEOUT, GuardReply(503, 0.5)]})
    p = probe_guard_host(G, get, lambda _: None)
    assert not p.ok and p.detail == "timeout, then http 503" and p.warm_s is None
    both, _ = script({G + "/health": [TIMEOUT]})
    assert probe_guard_host(G, both, lambda _: None).detail == "timeout, then timeout"


def test_health_body_ok_false_is_unhealthy() -> None:
    get, _ = script({G + "/health": [GuardReply(200, 0.1, {"ok": False})]})
    assert not probe_guard_host(G, get, lambda _: None).ok


def test_warm_call_failing_after_a_good_first_call_is_a_failure() -> None:
    get, _ = script({G + "/health": [HEALTH_OK, GuardReply(500, 0.2)]})
    p = probe_guard_host(G, get, lambda _: None)
    assert not p.ok and p.detail == "warm call http 500"


def test_vercel_info_failure_never_fails_the_probe() -> None:
    get, _ = script({G + "/health": [HEALTH_OK], G + "/vercel-info": [GuardReply(404, 0.1)]})
    p = probe_guard_host(G, get, lambda _: None)
    assert p.ok and not p.info_ok and p.served is None and p.region is None
    assert p.boot_s == 9.701  # the header still carries it
    junk = GuardReply(200, 0.1, {"served": True, "region": 3, "boot_error": "x", "boot_s": "nope"})
    get2, _ = script({G + "/health": [HEALTH_OK], G + "/vercel-info": [junk]})
    q = probe_guard_host(G, get2, lambda _: None)
    assert q.ok and q.served is None and q.region is None and q.boot_error == "x"
    assert q.boot_s == 9.701  # junk /vercel-info value ignored, header used


def test_adapt_get_marks_a_full_timeout_only() -> None:
    def plain(url: str, t: float) -> tuple[int, float, dict[str, Any]]:
        return (599, t, {}) if url.endswith("slow") else (599, 0.1, {})

    g = adapt_get(plain)
    assert g("http://x/slow", 5.0).timed_out and not g("http://x/fast", 5.0).timed_out


# ---- through the canary entry point


def test_run_records_guard_ok_and_boot_metrics_and_summary() -> None:
    pusher, fake, _ = make_pusher()
    get, _ = script({G + "/health": [COLD_OK, HEALTH_OK], G + "/vercel-info": [INFO]})
    rep = run(
        env(GUARD_URL=G),
        get=getter(GATEWAY_OK),
        guard_get=get,
        post=fake_post(),
        pusher=pusher,
        **NOSLEEP,
    )
    assert rep.status == "PASS" and rep.guard_state == "up" and not rep.warnings
    got = metric_names(fake.calls[0][1])
    assert got["guard_ok"] == 1 and got["guard_boot_seconds"] == 9.701
    assert got["guard_cold_start_seconds"] == 12.8 and got["guard_warm_latency_seconds"] == 0.3
    md = summary_md(rep)
    assert "### Guard (direct probe)" in md and "fresh instance: yes" in md
    assert "region iad1" in md and "boot seconds of the answering instance: 9.7" in md


def test_run_with_one_cold_start_timeout_stays_green_with_a_warning(
    capsys: Any,
) -> None:
    from breachlab.monitor.canary_cli import main

    get, _ = script({G + "/health": [TIMEOUT, COLD_OK, HEALTH_OK], G + "/vercel-info": [INFO]})
    code = main(
        [],
        env=env(GUARD_URL=G),
        get=getter(GATEWAY_OK),
        guard_get=get,
        post=fake_post(),
        **NOSLEEP,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "::warning title=guard warning::guard cold-start timeout" in out
    assert "### Warnings" in out and "PASS" in out


def test_run_with_two_guard_failures_is_red_and_guard_ok_is_zero() -> None:
    pusher, fake, _ = make_pusher()
    get, _ = script({G + "/health": [TIMEOUT]})
    rep = run(
        env(GUARD_URL=G),
        get=getter(GATEWAY_OK),
        guard_get=get,
        post=fake_post(guard="offline"),
        pusher=pusher,
        **NOSLEEP,
    )
    assert rep.failures == ["guard health: timeout, then timeout"]
    got = metric_names(fake.calls[0][1])
    assert got["guard_ok"] == 0 and "guard_boot_seconds" not in got
    assert got["guard_unavailable_total"] == 1


def test_no_guard_probe_without_a_provisioned_host() -> None:
    pusher, fake, _ = make_pusher()
    rep = run(
        env(GUARD_URL="https://placeholder.invalid"),
        get=getter(GATEWAY_OK),
        guard_get=lambda u, t: (_ for _ in ()).throw(AssertionError("must not be contacted")),
        post=fake_post(guard="offline"),
        pusher=pusher,
        **NOSLEEP,
    )
    assert rep.guard_probe is None and rep.degraded
    assert "guard_ok" not in metric_names(fake.calls[0][1])


def test_summary_without_a_probe_has_no_guard_section() -> None:
    assert "Guard (direct probe)" not in summary_md(Report())
