"""Degraded mode (ADR draft E2-canary-degraded-mode): guard offline is a known state, not a regression. No network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from breachlab.gateway.guards import RegexGuard
from breachlab.monitor import canary
from breachlab.monitor.canary import (
    ATTACKS,
    BENIGN,
    L2_DIRECT_ASK,
    TIER1_PLAN,
    is_guard_offline,
    probe_guard,
    run_canary,
    send_probe,
)
from breachlab.monitor.canary_cli import Health, Report, guard_provisioned, main, run, summary_md
from breachlab.monitor.canary_otlp import CANARY_METRICS
from breachlab.monitor.guard_probe import GuardReply
from tests.monitor.test_canary_cli import OK, fake_post, getter, make_pusher, metric_names

NOSLEEP: dict[str, Any] = {"sleep": lambda _: None, "probe_sleep": lambda _: None}
PLACEHOLDER = "https://placeholder.invalid"
GATEWAY_OK = {"http://t/api/health": [(200, {"ok": True, "guard": True})]}


def env(**kw: str) -> dict[str, str]:
    return {"BASE_URL": "http://t", "ADMIN_PASSWORD": "pw", **kw}


def test_tier1_plan_matches_the_real_regex_wall() -> None:
    """If someone edits the regex wall, tier 1 must be re-chosen: its expectations are deterministic."""
    assert {lv for lv, _, _ in TIER1_PLAN} == {1, 2}
    for level, text, is_attack in TIER1_PLAN:
        assert text in BENIGN + ATTACKS + [L2_DIRECT_ASK]
        if level == 2:
            assert (RegexGuard().classify(text)[0] == "UNSAFE") is is_attack, text
        else:
            assert not is_attack  # L1 has no guard: attacks are not deterministic there
    assert any(a for lv, _, a in TIER1_PLAN if lv == 2)
    assert any(not a for lv, _, a in TIER1_PLAN if lv == 2)


def test_is_guard_offline_only_for_503_guard() -> None:
    assert is_guard_offline(503, {"detail": "guard waking up, retry in a few seconds"})
    assert not is_guard_offline(503, {"detail": "the vault is busy, try again"})
    assert not is_guard_offline(500, {"detail": "guard waking up"})
    assert not is_guard_offline(503, {})
    assert not is_guard_offline(503, {"detail": ["guard"]})


def test_guard_provisioned() -> None:
    assert guard_provisioned("https://pseudo-harsh-guard.hf.space")
    assert guard_provisioned("http://g")
    for bad in [
        "",
        "  ",
        PLACEHOLDER,
        "https://placeholder.invalid/x",
        "not-a-url",
        "https://example.com",
    ]:
        assert not guard_provisioned(bad), bad


def test_send_probe_retries_busy_then_gives_up() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def post(url: str, body: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        calls.append(1)
        return (
            (503, {"detail": "the vault is busy"}) if len(calls) < 3 else (200, {"blocked": False})
        )

    assert send_probe("http://t", "s", 2, "hi", post, sleeps.append)[:2] == (
        200,
        {"blocked": False},
    )
    assert sleeps == [canary.BUSY_WAIT_S, 2 * canary.BUSY_WAIT_S]

    def always(u: str, b: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        return 503, {"detail": "the vault is busy"}

    assert send_probe("http://t", "s", 2, "hi", always, lambda _: None)[0] == 503
    n: list[int] = []

    def guard_503(u: str, b: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        n.append(1)
        return 503, {"detail": "guard waking up"}

    # a guard 503 is not retried
    assert send_probe("http://t", "s", 4, "hi", guard_503, lambda _: None)[0] == 503 and len(n) == 1


def test_probe_guard_states() -> None:
    def mk(status: int, body: dict[str, Any]) -> canary.Post:
        return lambda u, b, h: (status, body)

    def nosleep(_: float) -> None:
        return None

    assert probe_guard("http://t", "s", "k", mk(200, {"blocked": False}), nosleep) == ("up", "ok")
    st, why = probe_guard("http://t", "s", "k", mk(503, {"detail": "guard waking up"}), nosleep)
    assert st == "offline" and "guard waking up" in why
    assert probe_guard("http://t", "s", "k", mk(500, {}), nosleep) == ("error", "http 500")


def test_run_canary_wrapper_still_runs_the_twenty_probes() -> None:
    res = run_canary("http://t", "k", fake_post(), level=4, sleep=lambda _: None)
    assert len(res) == 20 and all(r.ok and r.level == 4 for r in res)


# ---- degraded branches of the CLI


def test_guard_offline_without_guard_host_is_degraded_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pusher, fake, _ = make_pusher()
    summary = tmp_path / "s.md"
    code = main(
        [],
        env=env(GUARD_URL=PLACEHOLDER, GITHUB_STEP_SUMMARY=str(summary)),
        get=getter(GATEWAY_OK),
        post=fake_post(guard="offline"),
        pusher=pusher,
        now=lambda: 1_700_000_000.0,
        **NOSLEEP,
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "::notice title=canary degraded::" in out
    md = summary.read_text()
    assert "DEGRADED" in md and "skipped (guard not up)" in md and "guard state: offline" in md
    assert "placeholder" in md  # the note names the placeholder host
    got = metric_names(fake.calls[0][1])
    assert got["canary_degraded_total"] == 1 and got["guard_up"] == 0
    assert got["canary_failures_total"] == 0 and got["guard_unavailable_total"] == 0
    assert got["canary_ok_total"] == 1 and got["canary_last_success_timestamp"] == 1_700_000_000.0
    assert set(got) <= set(CANARY_METRICS)


def test_placeholder_guard_host_is_never_contacted() -> None:
    urls: list[str] = []
    base = getter(GATEWAY_OK)

    def get(url: str, t: float) -> tuple[int, float, dict[str, Any]]:
        urls.append(url)
        return base(url, t)

    rep = run(env(GUARD_URL=PLACEHOLDER), get=get, post=fake_post(guard="offline"), **NOSLEEP)
    assert rep.ok and rep.degraded and rep.status == "DEGRADED"
    assert urls and all(u.startswith("http://t") for u in urls)
    assert len(rep.tier1) == len(TIER1_PLAN) and rep.probes == []


def test_unset_guard_url_is_degraded_too() -> None:
    rep = run(env(), get=getter(GATEWAY_OK), post=fake_post(guard="offline"), **NOSLEEP)
    assert rep.ok and rep.degraded and rep.guard_state == "offline" and rep.notes


def test_gateway_without_guard_flag_skips_the_level4_probe() -> None:
    seen: list[int] = []
    inner = fake_post(guard="offline")

    def post(url: str, body: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        if "level" in body:
            seen.append(body["level"])
        return inner(url, body, h)

    no_guard = {"http://t/api/health": [(200, {"ok": True, "guard": False})]}
    rep = run(env(), get=getter(no_guard), post=post, **NOSLEEP)
    assert rep.ok and rep.degraded and 4 not in seen and set(seen) == {1, 2}


def test_guard_up_runs_tier2_with_full_metrics() -> None:
    pusher, fake, _ = make_pusher()
    rep = run(
        env(GUARD_URL="http://g"),
        get=getter({"http://g/health": [OK], **GATEWAY_OK}),
        post=fake_post(),
        pusher=pusher,
        **NOSLEEP,
    )
    assert rep.status == "PASS" and rep.guard_state == "up" and not rep.degraded
    assert len(rep.tier1) == len(TIER1_PLAN) and len(rep.probes) == 20
    got = metric_names(fake.calls[0][1])
    assert got["guard_up"] == 1 and got["canary_degraded_total"] == 0
    assert "tier 2 probes (level 4, the 20 PRD probes) passed: 20/20" in summary_md(rep)


def test_guard_up_without_guard_url_in_canary_env_still_runs_tier2() -> None:
    """The workflow may lack GUARD_URL while the gateway has a live guard: the level-4 probe decides."""
    rep = run(env(), get=getter(GATEWAY_OK), post=fake_post(), **NOSLEEP)
    assert rep.status == "PASS" and len(rep.probes) == 20


def test_guard_offline_is_a_failure_when_a_guard_is_expected() -> None:
    pusher, fake, _ = make_pusher()
    rep = run(
        env(GUARD_URL="http://g"),
        get=getter({"http://g/health": [OK], **GATEWAY_OK}),
        post=fake_post(guard="offline"),
        pusher=pusher,
        **NOSLEEP,
    )
    assert not rep.ok and not rep.degraded
    assert rep.failures == [
        "guard offline (http 503: guard waking up, retry in a few seconds) but a guard is expected"
    ]
    got = metric_names(fake.calls[0][1])
    assert got["canary_failures_total"] == 1 and got["guard_up"] == 0
    assert got["canary_degraded_total"] == 0 and "canary_last_success_timestamp" not in got


def test_require_guard_flag_turns_degraded_into_failure() -> None:
    rep = run(
        env(CANARY_REQUIRE_GUARD="1"),
        get=getter(GATEWAY_OK),
        post=fake_post(guard="offline"),
        **NOSLEEP,
    )
    assert not rep.ok and not rep.degraded and "guard is expected" in rep.failures[0]


def test_guard_health_down_and_gateway_offline_reports_one_failure() -> None:
    rep = run(
        env(GUARD_URL="http://g"),
        get=getter({"http://g/health": [(503, {})], **GATEWAY_OK}),
        post=fake_post(guard="offline"),
        **NOSLEEP,
    )
    assert rep.failures == ["guard health: http 503, then http 503"]


def test_tier1_regression_is_real_even_when_guard_offline() -> None:
    inner = fake_post(guard="offline")

    def post(url: str, body: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        if body.get("level") == 2 and body["message"] == ATTACKS[0]:
            return 200, {"blocked": False, "leaked": False}  # regex wall broken
        return inner(url, body, h)

    pusher, fake, _ = make_pusher()
    rep = run(env(), get=getter(GATEWAY_OK), post=post, pusher=pusher, **NOSLEEP)
    assert rep.status == "FAIL" and rep.degraded
    assert rep.failures[0].startswith("tier1 L2 attack passed the guard")
    got = metric_names(fake.calls[0][1])
    assert got["canary_failures_total"] == 1 and got["canary_degraded_total"] == 1


def test_level4_non_guard_error_is_a_real_failure_and_skips_tier2() -> None:
    def post(url: str, body: dict[str, Any], h: dict[str, str]) -> tuple[int, dict[str, Any]]:
        if body.get("level") == 4:
            return 500, {}
        return fake_post()(url, body, h)

    rep = run(env(), get=getter(GATEWAY_OK), post=post, **NOSLEEP)
    assert rep.failures == ["guard probe at level 4: http 500"]
    assert rep.guard_state == "unknown" and rep.probes == []


def test_session_failure_is_reported_like_before() -> None:
    rep = run(env(), get=getter(GATEWAY_OK), post=lambda u, b, h: (500, {}), **NOSLEEP)
    assert rep.failures == ["http 500: '<session>'"] and rep.tier1 == []


def test_vault_busy_noise_is_retried_not_reported() -> None:
    sleeps: list[float] = []
    rep = run(
        env(),
        get=getter(GATEWAY_OK),
        post=fake_post(guard="offline", busy=2),
        sleep=lambda _: None,
        probe_sleep=sleeps.append,
    )
    assert rep.ok and canary.BUSY_WAIT_S in sleeps


def test_summary_degraded_and_failed_headings() -> None:
    md = summary_md(Report(degraded=True, guard_state="offline", notes=["no guard"]))
    assert "DEGRADED" in md and "### Notes" in md and "- no guard" in md
    fail = summary_md(Report(failures=["x"], health=[Health("g", False, None, None, "d")]))
    assert "FAIL" in fail


def test_every_canary_metric_is_emitted_and_ops_dashboard_uses_the_new_ones() -> None:
    emitted: set[str] = set()
    for guard, extra in (("up", {"GUARD_URL": "http://g"}), ("offline", {})):
        pusher, fake, _ = make_pusher()
        run(
            env(**extra),
            get=getter({"http://g/health": [OK], **GATEWAY_OK}),
            guard_get=lambda u, t: GuardReply(
                200, 0.5, {"ok": True}, {"x-guard-cold": "1", "x-guard-boot-s": "9.7"}
            ),
            post=fake_post(guard=guard),
            pusher=pusher,
            now=lambda: 1.0,
            **NOSLEEP,
        )
        emitted |= set(metric_names(fake.calls[0][1]))
    assert emitted | {"breachlab_otlp_ingest_test"} == set(CANARY_METRICS)
    root = Path(__file__).resolve().parents[2] / "dashboards"
    ops = json.loads((root / "ops.json").read_text())
    exprs = " ".join(t["expr"] for p in ops["panels"] for t in p["targets"])
    for name in ("guard_up", "canary_degraded_total"):
        assert name in exprs
