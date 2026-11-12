"""Direct guard probe for the canary (lane E3, ADR-0082): ``GUARD_URL/health`` plus the guard's own ``/vercel-info``.

The guard is a Vercel function whose instance is gone after a long idle gap; the first request then boots the model
for about 13 s (docs/lanes/B3-vercel-bench-cold.json). The gateway's own guard timeout is 8 s, so a player can see the
fail-closed 503 "guard waking up" (ADR-0050) while the canary, which waits longer, should not turn red for it.

What is probed, and what it reports:

* ``GET /health`` (no auth): ``guard_ok``. A first call that times out (or errors) is retried once after a short gap; if the
  retry answers, the run carries a WARNING (a cold-start timeout), not a failure. Two failures in a row are a failure.
* a second ``GET /health`` for the warm latency;
* ``GET /vercel-info`` (best effort, never a failure): ``instance_age_s``, ``served``, ``region``, ``boot_error``.
* ``boot_s``: ``/vercel-info`` ``boot_s`` when present; otherwise the guard's ``x-guard-boot-s`` response header (the same
  number, sent on every response; ``/vercel-info`` does not carry it today). ``x-guard-cold: 1`` marks the response that
  ran on a fresh instance, which is how a cold start is recognised without a clock.

Everything external is injected (``GuardGet``) so tests need no network.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

FIRST_TIMEOUT_S = 25.0  # a cold start measured 12.8 s; leave room for a slow Hub fetch
WARM_TIMEOUT_S = 20.0
INFO_TIMEOUT_S = 10.0
RETRY_GAP_S = 5.0


@dataclass(frozen=True)
class GuardReply:
    status: int
    elapsed_s: float
    body: dict[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    timed_out: bool = False


GuardGet = Callable[[str, float], GuardReply]


@dataclass(frozen=True)
class GuardProbe:
    ok: bool
    detail: str
    first_s: float | None  # time spent on the first /health call(s), retry included
    warm_s: float | None
    cold: bool | None  # the answering response ran on a fresh instance (x-guard-cold)
    boot_s: float | None
    instance_age_s: float | None = None
    served: int | None = None
    region: str | None = None
    boot_error: str | None = None
    info_ok: bool = False
    warnings: tuple[str, ...] = ()


def http_guard_get(
    url: str, timeout: float
) -> GuardReply:  # pragma: no cover - network, exercised by the live canary
    import httpx

    t0 = time.perf_counter()
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.TimeoutException:
        return GuardReply(599, time.perf_counter() - t0, timed_out=True)
    except httpx.HTTPError:
        return GuardReply(599, time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    try:
        body = r.json()
    except ValueError:
        body = {}
    return GuardReply(
        r.status_code,
        elapsed,
        body if isinstance(body, dict) else {},
        {k.lower(): v for k, v in r.headers.items()},
    )


def adapt_get(get: Callable[[str, float], tuple[int, float, dict[str, Any]]]) -> GuardGet:
    """Wrap the canary's plain ``Get`` (no headers); a 599 that used the whole timeout counts as a timeout."""

    def guard_get(url: str, timeout: float) -> GuardReply:
        status, elapsed, body = get(url, timeout)
        return GuardReply(status, elapsed, body, {}, timed_out=status == 599 and elapsed >= timeout)

    return guard_get


def _healthy(r: GuardReply) -> bool:
    return r.status == 200 and r.body.get("ok", True) is not False


def _label(r: GuardReply) -> str:
    return "timeout" if r.timed_out else f"http {r.status}"


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def probe_guard_host(
    url: str, get: GuardGet, sleep: Callable[[float], None] = time.sleep
) -> GuardProbe:
    """``url`` is the guard base URL without a trailing slash."""
    warnings: list[str] = []
    first = get(url + "/health", FIRST_TIMEOUT_S)
    first_s = first.elapsed_s
    answered = first
    if not _healthy(first):
        sleep(RETRY_GAP_S)
        retry = get(url + "/health", FIRST_TIMEOUT_S)
        first_s += retry.elapsed_s
        if not _healthy(retry):
            return GuardProbe(
                False, f"{_label(first)}, then {_label(retry)}", first_s, None, None, None
            )
        answered = retry
        if first.timed_out:
            warnings.append(
                f"guard cold-start timeout: the first /health call gave up after {first.elapsed_s:.0f} s, "
                f"the retry answered after {retry.elapsed_s:.1f} s (a fresh instance was booting; ADR-0082)"
            )
        else:
            warnings.append(
                f"guard /health failed once ({_label(first)}) and then answered after {retry.elapsed_s:.1f} s"
            )

    warm = get(url + "/health", WARM_TIMEOUT_S)
    if not _healthy(warm):
        return GuardProbe(
            False,
            f"warm call {_label(warm)}",
            first_s,
            warm.elapsed_s,
            None,
            None,
            warnings=tuple(warnings),
        )

    cold = answered.headers.get("x-guard-cold") == "1" if answered.headers else None
    boot_header = _num(answered.headers.get("x-guard-boot-s"))

    info = get(url + "/vercel-info", INFO_TIMEOUT_S)
    info_ok = info.status == 200 and bool(info.body)
    body = info.body if info_ok else {}
    boot_s = _num(body.get("boot_s")) or (boot_header if boot_header else None)
    served = body.get("served")
    region = body.get("region")
    boot_error = body.get("boot_error")
    return GuardProbe(
        True,
        "ok",
        first_s,
        warm.elapsed_s,
        cold,
        boot_s,
        instance_age_s=_num(body.get("instance_age_s")),
        served=served if isinstance(served, int) and not isinstance(served, bool) else None,
        region=region if isinstance(region, str) else None,
        boot_error=str(boot_error) if boot_error else None,
        info_ok=info_ok,
        warnings=tuple(warnings),
    )
