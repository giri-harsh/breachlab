"""Hourly keep-warm + canary entry point (ADR-0041): ``python -m breachlab.monitor.canary``.

Steps: (a) probe the guard directly (only when ``GUARD_URL`` is a real URL; ``monitor.guard_probe``: /health, /vercel-info,
guard_ok, boot seconds, a single cold-start timeout is a warning, ADR-0082) and health-check the gateway, a cold call first
and a warm call second so cold-start latency is recorded separately; (b) tier 1: probes at levels 1-2 (no guard
host needed); (c) one benign level-4 probe decides whether the guard is up. Up: tier 2 runs the 20 PRD probes at
level 4. Gateway answers 503 "guard waking up": the run is DEGRADED (exit 0, ``canary_degraded_total=1``,
``guard_up=0``, no ``canary_failures_total`` increment) unless a guard is expected (real ``GUARD_URL`` or
``CANARY_REQUIRE_GUARD=1``), which is a failure; (d) push metrics to Grafana Cloud over OTLP/HTTP. Real failures
exit non-zero so the workflow goes red. Everything external is injected so tests need no network (ADR draft E2-canary-degraded-mode).

Env: BASE_URL (default the Vercel gateway), GUARD_URL (optional; unset or ``placeholder.invalid`` means no guard
host yet), CANARY_REQUIRE_GUARD (1 = guard offline is a failure), GATEWAY_SECRET_KEY or ADMIN_PASSWORD (the HMAC
key, derived exactly like ``GatewaySettings.secret_key``), GRAFANA_OTLP_ENDPOINT / _INSTANCE_ID / _TOKEN,
CANARY_LEVEL (default 4). Exit codes: 0 all good or degraded, 1 canary failure, 2 not configured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from breachlab.monitor.canary import (
    PROBES,
    TIER1_GAP_S,
    TIER1_PLAN,
    Post,
    ProbeResult,
    http_post,
    probe_guard,
    run_plan,
    start_session,
)
from breachlab.monitor.canary_otlp import OtlpPusher, PushResult, pusher_from_env
from breachlab.monitor.guard_probe import (
    GuardGet,
    GuardProbe,
    adapt_get,
    http_guard_get,
    probe_guard_host,
)

DEFAULT_BASE_URL = "https://breachlab-api.vercel.app"
COLD_TIMEOUT_S = 120.0  # an idle free HF Space can need a minute or two to boot
WARM_TIMEOUT_S = 20.0

# (url, timeout_s) -> (status, elapsed_s, json body)
Get = Callable[[str, float], tuple[int, float, dict[str, Any]]]


def http_get(
    url: str, timeout: float
) -> tuple[int, float, dict[str, Any]]:  # pragma: no cover - network
    import httpx

    t0 = time.perf_counter()
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return 599, time.perf_counter() - t0, {}
    elapsed = time.perf_counter() - t0
    try:
        body = r.json()
    except ValueError:
        body = {}
    return r.status_code, elapsed, body if isinstance(body, dict) else {}


@dataclass(frozen=True)
class Health:
    name: str
    ok: bool
    cold_s: float | None
    warm_s: float | None
    detail: str
    guard_configured: bool | None = (
        None  # gateway /api/health "guard" flag (None when not reported)
    )


@dataclass
class Report:
    health: list[Health] = field(default_factory=list)
    tier1: list[ProbeResult] = field(default_factory=list)  # levels 1-2, no guard host needed
    probes: list[ProbeResult] = field(
        default_factory=list
    )  # tier 2: the 20 PRD probes (guard up only)
    failures: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    push: PushResult | None = None
    guard_state: str = "unknown"  # up | offline | unknown
    degraded: bool = False  # guard offline in the known state (no guard host); not a failure
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(
        default_factory=list
    )  # e.g. a single guard cold-start timeout; never a failure
    guard_probe: GuardProbe | None = (
        None  # direct GUARD_URL probe (only with a provisioned guard host)
    )

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        return "FAIL" if self.failures else "DEGRADED" if self.degraded else "PASS"


def derive_key(env: Mapping[str, str]) -> str:
    """Same rule as ``GatewaySettings.secret_key``; empty when neither variable is set."""
    if env.get("GATEWAY_SECRET_KEY"):
        return env["GATEWAY_SECRET_KEY"]
    if env.get("ADMIN_PASSWORD"):
        return hashlib.sha256(f"breachlab-gateway-v1|{env['ADMIN_PASSWORD']}".encode()).hexdigest()
    return ""


PLACEHOLDER_MARKERS = ("placeholder", ".invalid", "example.")


def guard_provisioned(url: str) -> bool:
    """True for a real guard host URL; empty or the ``https://placeholder.invalid`` stand-in means none yet."""
    u = url.strip().lower()
    return u.startswith(("http://", "https://")) and not any(m in u for m in PLACEHOLDER_MARKERS)


def _healthy(status: int, body: dict[str, Any]) -> bool:
    return status == 200 and body.get("ok", True) is not False


def check_health(
    name: str, url: str, get: Get, sleep: Callable[[float], None] = time.sleep
) -> Health:
    """Cold call (retried once: the Space may still be booting), then a warm call to measure steady latency."""
    status, cold, body = get(url, COLD_TIMEOUT_S)
    if not _healthy(status, body):
        first = status
        sleep(5.0)
        status, extra, body = get(url, COLD_TIMEOUT_S)
        cold += extra
        if not _healthy(status, body):
            return Health(name, False, cold, None, f"http {first}, then {status}")
    status, warm, body = get(url, WARM_TIMEOUT_S)
    if not _healthy(status, body):
        return Health(name, False, cold, warm, f"warm call http {status}")
    flag = body.get("guard")
    return Health(name, True, cold, warm, "ok", flag if isinstance(flag, bool) else None)


def run(
    env: Mapping[str, str],
    *,
    get: Get = http_get,
    guard_get: GuardGet | None = None,
    post: Post = http_post,
    pusher: OtlpPusher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.time,
    probe_sleep: Callable[[float], None] = time.sleep,
) -> Report:
    t0 = time.perf_counter()
    base = (env.get("BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    guard = (env.get("GUARD_URL") or "").strip().rstrip("/")
    level = int(env.get("CANARY_LEVEL") or 4)
    provisioned = guard_provisioned(guard)
    required = provisioned or (env.get("CANARY_REQUIRE_GUARD") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    rep = Report()
    if (
        guard_get is None
    ):  # headers (cold flag, boot seconds) need the real client; injected getters have none
        guard_get = http_guard_get if get is http_get else adapt_get(get)

    guard_down = False
    if provisioned:  # a placeholder host is never contacted: its absence is the known state
        gp = probe_guard_host(guard, guard_get, sleep)
        rep.guard_probe = gp
        rep.warnings += gp.warnings
        rep.health.append(Health("guard", gp.ok, gp.first_s, gp.warm_s, gp.detail))
        if not gp.ok:
            guard_down = True
            rep.failures.append(f"guard health: {gp.detail}")
    elif guard:
        rep.notes.append(f"GUARD_URL is a placeholder ({guard}): guard host not provisioned")
    gw = check_health("gateway", base + "/api/health", get, sleep)
    rep.health.append(gw)
    if not gw.ok:
        rep.failures.append(f"gateway health: {gw.detail}")
    else:
        _probe(
            rep, base, derive_key(env), post, level, required, guard_down, gw, probe_sleep, sleep
        )
    rep.duration_s = time.perf_counter() - t0

    if pusher is not None:
        _record(pusher, rep, guard_down, now())
        rep.push = pusher.flush()
    return rep


def _probe(
    rep: Report,
    base: str,
    key: str,
    post: Post,
    level: int,
    required: bool,
    guard_down: bool,
    gw: Health,
    probe_sleep: Callable[[float], None],
    sleep: Callable[[float], None],
) -> None:
    sid, status = start_session(base, key, post)
    if sid is None:
        rep.failures.append(f"http {status}: '<session>'")
        return
    # tier 1: levels 1-2, deterministic, needs no guard host
    rep.tier1 = run_plan(base, sid, key, post, TIER1_PLAN, None, probe_sleep, TIER1_GAP_S)
    rep.failures += [f"tier1 L{p.level} {p.detail}: {p.text[:60]!r}" for p in rep.tier1 if not p.ok]

    # tier 2 only when the guard answers
    if gw.guard_configured is False:
        state, why = "offline", "gateway reports no GUARD_URL"
    else:
        state, why = probe_guard(base, sid, key, post, sleep)
    if state == "error":
        rep.failures.append(f"guard probe at level 4: {why}")
        return
    if state == "offline":
        rep.guard_state = "offline"
        if required:
            if not guard_down:  # a failed guard health check already reported the same outage
                rep.failures.append(f"guard offline ({why}) but a guard is expected")
        else:
            rep.degraded = True
            rep.notes.append(
                f"guard host not provisioned or unreachable ({why}): tier 2 (level {level}) skipped"
            )
        return
    rep.guard_state = "up"
    rep.probes = run_plan(
        base, sid, key, post, [(level, t, a) for t, a in PROBES], None, probe_sleep
    )
    rep.failures += [f"{p.detail}: {p.text[:60]!r}" for p in rep.probes if not p.ok]


def _record(p: OtlpPusher, rep: Report, guard_down: bool, ts: float) -> None:
    for h in rep.health:
        if h.cold_s is not None and h.ok:
            p.gauge(f"{h.name}_cold_start_seconds", h.cold_s)
        if h.warm_s is not None and h.ok:
            p.gauge(f"{h.name}_warm_latency_seconds", h.warm_s)
    p.counter("guard_unavailable_total", 1 if guard_down else 0)
    if rep.guard_probe is not None:
        p.gauge("guard_ok", 1 if rep.guard_probe.ok else 0)
        if rep.guard_probe.boot_s is not None:
            p.gauge("guard_boot_seconds", rep.guard_probe.boot_s)
    if rep.guard_state != "unknown":
        p.gauge("guard_up", 1 if rep.guard_state == "up" else 0)
    p.counter("canary_degraded_total", 1 if rep.degraded else 0)
    p.counter("canary_failures_total", len(rep.failures))
    p.counter("canary_ok_total", 1 if rep.ok else 0)
    p.gauge("canary_last_run_failures", len(rep.failures))
    p.gauge("canary_duration_seconds", rep.duration_s)
    if rep.ok:
        p.gauge("canary_last_success_timestamp", ts)


def _yn(v: bool | None) -> str:
    return "unknown" if v is None else "yes" if v else "no"


def _s(v: float | None) -> str:
    return "unknown" if v is None else f"{v:.1f}"


def summary_md(rep: Report) -> str:
    heading = {
        "PASS": "PASS",
        "DEGRADED": "DEGRADED (guard offline, tier 1 passed)",
        "FAIL": "FAIL",
    }[rep.status]
    lines = [f"## Canary: {heading}", ""]
    lines += ["| target | status | cold start (s) | warm (s) |", "|---|---|---|---|"]
    for h in rep.health:
        c = f"{h.cold_s:.2f}" if h.cold_s is not None else "-"
        w = f"{h.warm_s:.2f}" if h.warm_s is not None else "-"
        lines.append(f"| {h.name} | {h.detail} | {c} | {w} |")
    n_ok = sum(p.ok for p in rep.probes)
    t1_ok = sum(p.ok for p in rep.tier1)
    tier2 = f"passed: {n_ok}/{len(rep.probes)}" if rep.probes else "skipped (guard not up)"
    lines += [
        "",
        f"- tier 1 probes (levels 1-2, no guard needed) passed: {t1_ok}/{len(rep.tier1)}",
        f"- tier 2 probes (level 4, the 20 PRD probes) {tier2}",
        f"- guard state: {rep.guard_state}",
        f"- duration: {rep.duration_s:.1f} s",
    ]
    if rep.push is not None:
        lines.append(
            f"- Grafana OTLP push: {'ok' if rep.push.ok else 'FAILED'} (http {rep.push.status})"
        )
    else:
        lines.append("- Grafana OTLP push: skipped (no credentials)")
    gp = rep.guard_probe
    if gp is not None:
        lines += [
            "",
            "### Guard (direct probe)",
            f"- guard_ok: {'yes' if gp.ok else 'NO'} ({gp.detail})",
            f"- first request answered on a fresh instance: {_yn(gp.cold)}",
            f"- boot seconds of the answering instance: {_s(gp.boot_s)}",
            f"- /vercel-info: instance age {_s(gp.instance_age_s)} s, requests served "
            f"{gp.served if gp.served is not None else 'unknown'}, region {gp.region or 'unknown'}"
            + ("" if gp.info_ok else " (unavailable)"),
        ]
        if gp.boot_error:
            lines.append(f"- boot_error: {gp.boot_error}")
    if rep.warnings:
        lines += ["", "### Warnings"] + [f"- {w}" for w in rep.warnings]
    if rep.notes:
        lines += ["", "### Notes"] + [f"- {n}" for n in rep.notes]
    if rep.failures:
        lines += ["", "### Failures"] + [f"- {f}" for f in rep.failures]
    return "\n".join(lines) + "\n"


def ingest_test(pusher: OtlpPusher, ts: float) -> PushResult:
    """Push ``breachlab_otlp_ingest_test`` (value = unix timestamp) and return the HTTP outcome."""
    pusher.gauge("breachlab_otlp_ingest_test", ts)
    return pusher.flush()


def main(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    **kw: Any,
) -> int:
    ap = argparse.ArgumentParser(prog="python -m breachlab.monitor.canary")
    ap.add_argument(
        "--ingest-test", action="store_true", help="only push the OTLP ingest test metric"
    )
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    e = os.environ if env is None else env
    pusher = kw.pop("pusher", None) or pusher_from_env(e)

    if args.ingest_test:
        if pusher is None:
            print("GRAFANA_OTLP_* not set: cannot push")
            return 2
        res = ingest_test(pusher, kw.get("now", time.time)())
        print(f"otlp ingest test: ok={res.ok} http={res.status} ({res.detail})")
        return 0 if res.ok else 1

    if not derive_key(e):
        print("GATEWAY_SECRET_KEY or ADMIN_PASSWORD is required to sign canary requests")
        return 2
    rep = run(e, pusher=pusher, **kw)
    md = summary_md(rep)
    print(md)
    if args.json:
        print(json.dumps({"ok": rep.ok, "failures": rep.failures, "duration_s": rep.duration_s}))
    if rep.degraded and rep.ok:
        print(
            "::notice title=canary degraded::guard host offline (known state, ADR-0063): "
            "tier 1 (levels 1-2) passed, tier 2 (level 4) skipped; canary_degraded_total=1, guard_up=0"
        )
    for w in rep.warnings:
        print(f"::warning title=guard warning::{w}")
    if rep.push is not None and not rep.push.ok:
        print(
            f"::warning::OTLP push failed (http {rep.push.status}); metrics for this run are missing"
        )
    summary_path = e.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md)
    return 0 if rep.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
