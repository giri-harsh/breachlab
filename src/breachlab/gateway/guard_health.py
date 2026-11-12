"""Cached three-state view of the guard service for ``/api/health`` (``guard_state``).

Never used on the message path. States:

* ``ok``      the guard answered ``GET /health`` within ``timeout`` (2 s) with the champion loaded.
* ``warming`` it did not (a serverless guard that has been idle is cold and needs about 10 to 15 s to boot). The probe
  itself reaches the host, and a non-blocking warm (``GET /health`` with ``warm_timeout`` = 20 s, in a background
  thread, at most one in flight and one per ``warm_gap`` seconds) makes sure the boot is not abandoned.
* ``down``    only when the last successful contact is older than ``down_after_s`` (10 min, or never in this process)
  AND a warm attempt with the long timeout has failed. A cold guard is therefore never reported as down.

``ok()`` stays for backward compatibility (``guard_ok``: true only for ``ok``). Results are cached (``ttl`` seconds when
ok, ``fail_ttl`` otherwise), only one thread probes at a time (concurrent callers get the cached state), and a missing
URL or a placeholder host (``*.invalid``) is ``down`` without touching the network. Evidence from the message path and
from ``GET /api/warm`` is shared through ``attach(HttpGuard)``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

log = logging.getLogger("breachlab")

GuardState = Literal["ok", "warming", "down"]
PROBE_TIMEOUT_S = 2.0
WARM_TIMEOUT_S = 20.0
DOWN_AFTER_S = 600.0


def placeholder_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return not host or host == "invalid" or host.endswith(".invalid")


def _spawn_thread(fn: Callable[[], None]) -> None:  # pragma: no cover - thread plumbing
    threading.Thread(target=fn, name="guard-warm", daemon=True).start()


class GuardProbe:
    def __init__(
        self,
        url: str,
        *,
        token: str = "",
        ttl: float = 15.0,
        fail_ttl: float = 2.0,
        timeout: float = PROBE_TIMEOUT_S,
        warm_timeout: float = WARM_TIMEOUT_S,
        down_after_s: float = DOWN_AFTER_S,
        warm_gap: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        get: Callable[..., httpx.Response] | None = None,
        spawn: Callable[[Callable[[], None]], None] = _spawn_thread,
    ) -> None:
        self.url = url.rstrip("/")
        self._token, self.ttl, self.fail_ttl = token, ttl, fail_ttl
        self.timeout, self.warm_timeout = min(timeout, PROBE_TIMEOUT_S), warm_timeout
        self.down_after_s, self.warm_gap = down_after_s, warm_gap
        self._clock, self._get, self._spawn = clock, get or httpx.get, spawn
        self._configured = bool(self.url) and not placeholder_url(self.url)
        self._state: GuardState = "warming" if self._configured else "down"
        self._next = 0.0
        self._lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._last_ok: float | None = None
        self._warm_failed = False
        self._warm_inflight = False
        self._warm_started: float | None = None
        self._evidence: list[Any] = []

    def attach(self, guard: Any) -> None:
        """Share the message path's and ``/api/warm``'s evidence (``HttpGuard.last_ok_age`` / ``warm_failed``)."""
        self._evidence.append(guard)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _healthy(self, r: httpx.Response) -> bool:
        if r.status_code != 200:
            return False
        try:
            body: Any = r.json()
        except ValueError:
            return False
        if not isinstance(body, dict) or body.get("ok") is not True:
            return False
        models = body.get("models")
        return not (isinstance(models, dict) and not models.get("champion"))

    def _probe(self) -> bool:
        try:
            r = self._get(
                f"{self.url}/health",
                headers=self._headers(),
                timeout=self.timeout,
                follow_redirects=False,
            )
            return self._healthy(r)
        except httpx.HTTPError as exc:
            log.warning("guard health probe failed (%s)", type(exc).__name__)
            return False

    def _warm_worker(self) -> None:
        ok = False
        try:
            r = self._get(
                f"{self.url}/health",
                headers=self._headers(),
                timeout=self.warm_timeout,
                follow_redirects=False,
            )
            ok = self._healthy(r)
        except httpx.HTTPError as exc:
            log.warning("guard warm attempt failed (%s)", type(exc).__name__)
        with self._lock:
            self._warm_inflight = False
            if ok:
                self._last_ok, self._warm_failed = self._clock(), False
            else:
                self._warm_failed = True

    def _start_warm(self) -> None:
        """Non-blocking: at most one attempt in flight and one per ``warm_gap`` seconds."""
        now = self._clock()
        with self._lock:
            if self._warm_inflight:
                return
            if self._warm_started is not None and now - self._warm_started < self.warm_gap:
                return
            self._warm_inflight, self._warm_started = True, now
        self._spawn(self._warm_worker)

    def _since_ok(self) -> float | None:
        """Seconds since the last successful contact from any source, or None when there was none."""
        now = self._clock()
        ages = [] if self._last_ok is None else [now - self._last_ok]
        for g in self._evidence:
            age = g.last_ok_age()
            if age is not None:
                ages.append(age)
        return min(ages) if ages else None

    def _warm_has_failed(self) -> bool:
        return self._warm_failed or any(g.warm_failed for g in self._evidence)

    def _evaluate(self) -> GuardState:
        if not self._configured:
            return "down"
        if self._probe():
            with self._lock:
                self._last_ok, self._warm_failed = self._clock(), False
            for g in self._evidence:
                g.warm_failed = False
            return "ok"
        self._start_warm()
        since = self._since_ok()
        stale = since is None or since > self.down_after_s
        return "down" if stale and self._warm_has_failed() else "warming"

    def state(self) -> GuardState:
        """Cached state; refreshes by one caller at a time, without making the others wait."""
        if self._clock() < self._next:
            return self._state
        # The probe runs under its own lock (a 2 s network wait); concurrent callers get the cached state at once.
        if not self._probe_lock.acquire(blocking=False):
            return self._state
        try:
            if self._clock() >= self._next:
                self._state = self._evaluate()
                self._next = self._clock() + (self.ttl if self._state == "ok" else self.fail_ttl)
            return self._state
        finally:
            self._probe_lock.release()

    def ok(self) -> bool:
        return self.state() == "ok"
