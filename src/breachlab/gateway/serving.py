"""Champion hot-swap (PRD 10.7): which guard version the gateway routes to and reports.

The MLflow registry alias ``champion`` of model ``breachlab-guard`` is the source of truth. Vercel instances are
short-lived and frozen between requests, so the primary mechanism is *poll-on-request with a TTL cache*: the
first request after ``ttl`` seconds refreshes the alias (one refresher at a time, everyone else keeps using the
current reference), a failed poll keeps the old champion. ``ServingState.start_background`` is an optional
thread variant for long-lived hosts (Docker).

A request takes ``ServingState.snapshot()`` once and uses that immutable ``ServingRef`` from guard call to log
line, so a swap in the middle of a request never mixes versions and never drops it (the old reference simply
finishes). ``serving_version()`` is the single function every consumer (guards, telemetry, attempts, leaderboard)
uses to name the version being served.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

log = logging.getLogger("breachlab")

MODEL = "breachlab-guard"
DEFAULT_TTL_S = 30.0
UNKNOWN = "unknown"


@dataclass(frozen=True)
class ServingRef:
    """Immutable description of what the gateway serves right now."""

    alias: str = "champion"  # guard alias sent to the Space for L4/L5
    version: str = UNKNOWN  # registry version, e.g. "2"; "unknown" until the first successful poll
    hf_repo: str | None = None
    kernel: str | None = None
    checked_at: float | None = None  # clock() of the last successful poll

    @property
    def label(self) -> str:
        """Human/leaderboard tag, e.g. ``v2``; ``unknown`` before the first poll."""
        return self.version if self.version == UNKNOWN else f"v{self.version}"


class ChampionSource(Protocol):
    def fetch(self) -> ServingRef:
        """Return the current champion or raise on any failure."""
        ...


class RegistryChampionSource:
    """Reads ``champion`` through an ``mlflow.MlflowClient``-like object (``get_model_version_by_alias``)."""

    def __init__(self, client: Any, model: str = MODEL) -> None:
        self.client, self.model = client, model

    def fetch(self) -> ServingRef:
        mv = self.client.get_model_version_by_alias(self.model, "champion")
        tags = getattr(mv, "tags", None) or {}
        return ServingRef(
            version=str(mv.version), hf_repo=tags.get("hf_repo"), kernel=tags.get("kernel")
        )


class RestChampionSource:
    """MLflow REST (no mlflow import: the Vercel bundle is tiny). Basic or bearer auth from the env."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        model: str = MODEL,
        timeout: float = 3.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base = tracking_uri.rstrip("/")
        self.model, self.timeout = model, timeout
        self.auth = (username, password) if username and password else None
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = client

    def fetch(self) -> ServingRef:
        get = self.client.get if self.client is not None else httpx.get
        r = get(
            f"{self.base}/api/2.0/mlflow/registered-models/alias",
            params={"name": self.model, "alias": "champion"},
            headers=self.headers,
            auth=self.auth,
            timeout=self.timeout,
        )
        r.raise_for_status()
        mv = r.json()["model_version"]
        raw = mv.get("tags") or []
        tags = {t["key"]: t["value"] for t in raw} if isinstance(raw, list) else dict(raw)
        return ServingRef(
            version=str(mv["version"]), hf_repo=tags.get("hf_repo"), kernel=tags.get("kernel")
        )


def source_from_env(env: dict[str, str] | None = None) -> ChampionSource | None:
    """A REST source when ``MLFLOW_TRACKING_URI`` is set, else None (the gateway then reports guard versions)."""
    e = os.environ if env is None else env
    uri = e.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    return RestChampionSource(
        uri,
        username=e.get("MLFLOW_TRACKING_USERNAME"),
        password=e.get("MLFLOW_TRACKING_PASSWORD"),
        token=e.get("MLFLOW_TRACKING_TOKEN"),
    )


class ServingState:
    """Holds the current ``ServingRef``; swaps it atomically (single attribute rebind under a lock)."""

    def __init__(
        self,
        source: ChampionSource | None = None,
        *,
        ttl: float = DEFAULT_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        initial: ServingRef | None = None,
        on_swap: Callable[[ServingRef, ServingRef], None] | None = None,
    ) -> None:
        self.source, self.ttl, self.clock, self.on_swap = source, max(ttl, 0.0), clock, on_swap
        self._ref = initial or ServingRef()
        self._lock = threading.Lock()  # guards the swap itself
        self._refresh_lock = threading.Lock()  # one refresher at a time
        self._next_poll = 0.0  # clock() value at which the next poll is allowed
        self.failures = 0  # consecutive failed polls (resets on success)
        self.last_error: str | None = None
        self.polls = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- reading
    def current(self) -> ServingRef:
        """The reference as of now, without polling."""
        return self._ref

    def snapshot(self) -> ServingRef:
        """Poll-on-request: refresh if the TTL has elapsed, then return the (immutable) current reference."""
        if self.source is not None and self.clock() >= self._next_poll:
            self.refresh()
        return self._ref

    def version(self) -> str:
        return self.snapshot().label

    # ---- writing
    def refresh(self, force: bool = False) -> bool:
        """Poll the source once. Returns True when the champion changed. Never raises; keeps the old one on error."""
        if self.source is None:
            return False
        if not self._refresh_lock.acquire(blocking=force):
            return False  # another request is already refreshing; keep serving the current ref
        try:
            if not force and self.clock() < self._next_poll:
                return False
            self.polls += 1
            # Set first so a slow or failing registry is retried at most once per TTL.
            self._next_poll = self.clock() + self.ttl
            try:
                fetched = self.source.fetch()
            except Exception as exc:  # noqa: BLE001 - any registry/network failure keeps the old champion
                self.failures += 1
                self.last_error = type(exc).__name__
                log.warning(
                    "champion poll failed (%s); keeping %s", self.last_error, self._ref.label
                )
                return False
            self.failures, self.last_error = 0, None
            return self._swap(replace(fetched, checked_at=self.clock()))
        finally:
            self._refresh_lock.release()

    def _swap(self, new: ServingRef) -> bool:
        with self._lock:
            old = self._ref
            self._ref = new  # atomic rebind: in-flight requests keep their old snapshot
        changed = (old.version, old.alias) != (new.version, new.alias)
        if changed:
            log.info("champion swap %s -> %s", old.label, new.label)
            if self.on_swap is not None:
                try:
                    self.on_swap(old, new)
                except Exception:  # noqa: BLE001 - observers must not break serving
                    log.exception("on_swap observer failed")
        return changed

    # ---- optional background variant (long-lived hosts only)
    def start_background(self, interval: float | None = None) -> None:
        if self._thread is not None or self.source is None:
            return
        wait = interval if interval is not None else self.ttl

        def loop() -> None:
            while not self._stop.wait(wait):
                self.refresh(force=True)

        self._thread = threading.Thread(target=loop, name="champion-poller", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._stop.clear()


_default = ServingState()


def set_default(state: ServingState) -> None:
    global _default
    _default = state


def get_default() -> ServingState:
    return _default


def resolve_label(ref: ServingRef, fallback: str | None = None) -> str:
    """Name for ``ref``; ``fallback`` (e.g. the version the guard Space reported) only while it is unknown."""
    if ref.version == UNKNOWN and fallback:
        return fallback
    return ref.label


def serving_version(state: ServingState | None = None, fallback: str | None = None) -> str:
    """THE function naming the guard version being served (guards, telemetry, attempts, leaderboard).

    Polls on request when the TTL elapsed. A request that already holds a ``snapshot()`` passes through
    ``resolve_label`` on that snapshot instead, so one request never mixes two versions.
    """
    return resolve_label((state or _default).snapshot(), fallback)
