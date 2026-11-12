"""Input/output guards behind one protocol: regex (L2), HTTP guard service (L3-5), and a null guard."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from breachlab.serve.baselines import regex_classify

Kind = Literal["input", "output"]


@dataclass(frozen=True)
class GuardOutcome:
    label: str  # SAFE | UNSAFE
    category: str
    confidence: float
    version: str
    latency_ms: float = 0.0
    p_unsafe: float | None = None  # CONTRACTS s1; None only from servers older than the contract


class Guard(Protocol):
    def score(self, text: str, kind: Kind = "input", alias: str = "champion") -> GuardOutcome: ...


class RegexGuard:
    """Level 2's keyword wall. ``patterns`` ({category: [regex]}, first match wins) defaults to the
    ``keyword_wall`` of configs/levels.yaml, else the eval baseline's pattern list (``serve.baselines``)."""

    version = "regex-v1"

    def __init__(self, patterns: dict[str, list[str]] | None = None) -> None:
        if patterns is None:
            from breachlab.gateway.levels import default_levels

            patterns = default_levels().keyword_wall
        self._compiled = (
            {c: [re.compile(p, re.IGNORECASE) for p in ps] for c, ps in patterns.items()}
            if patterns is not None
            else None
        )
        if patterns is not None:
            digest = hashlib.sha256(repr(sorted(patterns.items())).encode()).hexdigest()[:6]
            self.version = f"regex-{digest}"

    def classify(self, text: str) -> tuple[str, str]:
        if self._compiled is None:
            return regex_classify(text)
        for cat, pats in self._compiled.items():
            if any(p.search(text) for p in pats):
                return "UNSAFE", cat
        return "SAFE", "none"

    def score(self, text: str, kind: Kind = "input", alias: str = "champion") -> GuardOutcome:
        label, cat = self.classify(text)
        return GuardOutcome(label, cat, 1.0 if label == "UNSAFE" else 0.0, self.version)


class GuardUnavailableError(RuntimeError):
    """The guard did not answer. ``warming`` is True while it is probably just booting (ADR-0085): the caller
    answers 202 instead of 503 and the UI retries; after the grace window it is a plain failure (fail closed)."""

    def __init__(self, message: str = "", warming: bool = False) -> None:
        super().__init__(message)
        self.warming = warming


LABELS = {"SAFE", "UNSAFE"}


def parse_guard_response(d: dict[str, Any]) -> GuardOutcome:
    """Parse a CONTRACTS s1 response. Tolerates servers lacking ``p_unsafe`` (confidence is p_unsafe by contract).

    Raises ``KeyError``/``ValueError``/``TypeError`` on a malformed body so callers fail closed.
    """
    label = str(d["label"])
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}")
    conf_raw = d.get("confidence")
    p_raw = d.get("p_unsafe")
    if p_raw is None and conf_raw is None:
        raise KeyError("p_unsafe")
    p = float(p_raw if p_raw is not None else conf_raw)  # type: ignore[arg-type]
    conf = float(conf_raw if conf_raw is not None else p)
    if not (0.0 <= p <= 1.0) or not (0.0 <= conf <= 1.0):
        raise ValueError("probability out of range")
    return GuardOutcome(
        label,
        str(d["category"]),
        conf,
        str(d["version"]),
        float(d.get("latency_ms") or 0.0),
        p,
    )


class HttpGuard:
    """Calls the guard service (Vercel function): POST {url}/score {text, kind, alias}.

    ``token`` (default env ``GUARD_TOKEN``) is sent as ``Authorization: Bearer`` and is never logged. ``client`` is an
    optional ``httpx.Client`` (tests, connection reuse); otherwise ``httpx.post`` is used. Any transport error, timeout,
    non-2xx or malformed body raises ``GuardUnavailableError`` after ``retries`` retries: callers fail closed.

    Cold starts (ADR-0085): a serverless guard instance is gone after about 10 minutes idle and its first answer takes
    about 13 s. When this guard has not answered for ``cold_after_s`` (or never), calls use ``cold_timeout`` instead of
    ``timeout``. A failure while cold is reported as ``warming`` for ``warm_grace_s`` after the first failure, so the
    gateway answers 202 and the UI retries; a guard that still fails after the grace window is down (503, fail closed).
    ``warm()`` pings ``/health`` (which boots the model) and is what ``GET /api/warm`` calls.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 1.5,
        retries: int = 1,
        token: str | None = None,
        client: httpx.Client | None = None,
        cold_timeout: float = 20.0,
        cold_after_s: float = 120.0,
        warm_grace_s: float = 60.0,
        warm_cache_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url, self.timeout, self.retries = url.rstrip("/"), timeout, retries
        self._token = token if token is not None else os.environ.get("GUARD_TOKEN", "")
        self.client = client
        self.cold_timeout = max(cold_timeout, timeout)
        self.cold_after_s, self.warm_grace_s, self.warm_cache_s = (
            cold_after_s,
            warm_grace_s,
            warm_cache_s,
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._last_ok: float | None = None
        self._first_fail: float | None = None
        self.warm_failed = (
            False  # the last /health warm attempt (cold timeout) failed; read by GuardProbe
        )
        self._warm_at: float | None = None
        self._warm_result: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def is_cold(self) -> bool:
        """True when this guard has not answered recently (or ever): its next call may have to wait for a boot."""
        last = self._last_ok
        return last is None or (self._clock() - last) > self.cold_after_s

    def last_ok_age(self) -> float | None:
        """Seconds since this guard last answered, None if it never did in this process."""
        last = self._last_ok
        return None if last is None else self._clock() - last

    def _ok(self) -> None:
        with self._lock:
            self._last_ok, self._first_fail = self._clock(), None
            self.warm_failed = False

    def _failed(self, was_cold: bool) -> bool:
        """Record a failure; True while it still counts as 'warming' (cold when it started, inside the grace window)."""
        now = self._clock()
        with self._lock:
            if self._first_fail is None:
                self._first_fail = now
            return was_cold and (now - self._first_fail) < self.warm_grace_s

    def score(self, text: str, kind: Kind = "input", alias: str = "champion") -> GuardOutcome:
        last: Exception | None = None
        extra: dict[str, Any] = {"headers": self._headers()} if self._token else {}
        was_cold = self.is_cold()
        timeout = self.cold_timeout if was_cold else self.timeout
        for _ in range(self.retries + 1):
            try:
                post = self.client.post if self.client is not None else httpx.post
                r = post(
                    f"{self.url}/score",
                    json={"text": text, "kind": kind, "alias": alias},
                    timeout=timeout,
                    **extra,
                )
                r.raise_for_status()
                out = parse_guard_response(r.json())
                self._ok()
                return out
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                last = exc
        raise GuardUnavailableError(
            type(last).__name__ if last else "guard failed", warming=self._failed(was_cold)
        )

    def warm(self) -> dict[str, Any]:
        """GET {url}/health with the cold timeout. Answers ``{"status": "warm"|"warming"|"down", "ms": int}``.

        Results are cached for ``warm_cache_s`` so a flood of page loads costs the guard at most a few pings a minute.
        """
        with self._lock:
            now = self._clock()
            if self._warm_at is not None and (now - self._warm_at) < self.warm_cache_s:
                return dict(self._warm_result)
        t0 = time.perf_counter()
        was_cold = self.is_cold()
        status = "warm"
        try:
            get = self.client.get if self.client is not None else httpx.get
            r = get(f"{self.url}/health", headers=self._headers(), timeout=self.cold_timeout)
            r.raise_for_status()
            self._ok()
        except httpx.HTTPError:
            status = "warming" if self._failed(was_cold) else "down"
            self.warm_failed = True
        result = {"status": status, "ms": round((time.perf_counter() - t0) * 1000)}
        with self._lock:
            self._warm_at, self._warm_result = self._clock(), result
        return dict(result)
