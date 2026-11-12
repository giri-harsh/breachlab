"""LLM client protocol, Groq/Gemini HTTP clients with exponential backoff, and a mock for tests."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

import httpx

T = TypeVar("T")


class LLMClient(Protocol):
    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> str: ...


class LLMError(RuntimeError):
    pass


def with_backoff(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry fn on LLMError/httpx errors with exponential backoff (base * 2**attempt)."""
    for attempt in range(retries):
        try:
            return fn()
        except (LLMError, httpx.HTTPError, ValueError):
            if attempt == retries - 1:
                raise
            sleep(base * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


class MockLLM:
    """Deterministic fake: returns a transform of the prompt's last line."""

    def __init__(self, fn: Callable[[str], str] | None = None) -> None:
        self.fn = fn or (lambda p: "MOCK: " + p.strip().splitlines()[-1])
        self.calls: list[str] = []

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        return self.fn(prompt)


class GroqClient:  # pragma: no cover - network
    def __init__(self, model: str = "openai/gpt-oss-120b", retries: int = 5) -> None:
        self.model = model
        self.retries = retries
        self.key = os.environ["GROQ_API_KEY"]

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        def call() -> str:
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            if r.status_code in (429, 500, 502, 503):
                raise LLMError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])

        return with_backoff(call, retries=self.retries)


class GeminiClient:  # pragma: no cover - network
    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model
        self.key = os.environ["GEMINI_API_KEY"]

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        def call() -> str:
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
                },
                timeout=60,
            )
            if r.status_code in (429, 500, 502, 503):
                raise LLMError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return str(r.json()["candidates"][0]["content"]["parts"][0]["text"])

        return with_backoff(call)


class RoundRobinLLM:
    """Split load across several clients (Groq + Gemini), PRD 15."""

    def __init__(self, clients: list[LLMClient]) -> None:
        self.clients = clients
        self._i = 0

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        c = self.clients[self._i % len(self.clients)]
        self._i += 1
        return c.complete(prompt, temperature=temperature, max_tokens=max_tokens)


class GeminiPool:  # pragma: no cover - network
    """Rotate over several Gemini models (each has its own free-tier quota); cool a model down on 429.

    Per-minute 429s cool a model for 65 s; per-day quota errors disable it for an hour. When every model
    is cooling the caller sleeps until the earliest is back, so the pool never gives up on rate limits.
    """

    MODELS = (
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    )

    def __init__(self, models: tuple[str, ...] | None = None) -> None:
        self.models = models or self.MODELS
        self.key = os.environ["GEMINI_API_KEY"]
        self._until = dict.fromkeys(self.models, 0.0)
        self._strikes = dict.fromkeys(self.models, 0)
        self._i = 0
        self._lock = threading.Lock()

    def _pick(self) -> tuple[str | None, float]:
        with self._lock:
            now = time.time()
            for _ in range(len(self.models)):
                m = self.models[self._i % len(self.models)]
                self._i += 1
                if self._until[m] <= now:
                    return m, 0.0
            return None, min(self._until.values()) - now

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        errors = 0
        while True:
            model, wait = self._pick()
            if model is None:
                if wait > 30:  # let the caller fail over to another provider instead of blocking
                    raise LLMError("all Gemini models cooling down")
                time.sleep(max(wait, 1.0))
                continue
            try:
                r = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": self.key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens,
                        },
                    },
                    timeout=90,
                )
            except httpx.HTTPError:
                errors += 1
                if errors > 8:
                    raise
                time.sleep(2 ** min(errors, 5))
                continue
            if r.status_code == 429:
                with self._lock:  # escalate on repeated 429s: 65 s, 130 s, ... capped at 1 h
                    self._strikes[model] += 1
                    self._until[model] = time.time() + min(
                        3600, 65 * 2 ** (self._strikes[model] - 1)
                    )
                continue
            if r.status_code == 404:  # model retired for this key: never pick it again
                with self._lock:
                    self._until[model] = float("inf")
                if all(v == float("inf") for v in self._until.values()):
                    raise LLMError("no Gemini model available")
                continue
            if r.status_code in (500, 502, 503, 504):
                with self._lock:
                    self._until[model] = time.time() + 30
                errors += 1
                if errors > 12:
                    raise LLMError(f"HTTP {r.status_code}")
                continue
            r.raise_for_status()
            with self._lock:
                self._strikes[model] = 0
            try:
                return str(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except (KeyError, IndexError, ValueError):
                return ""  # safety-blocked / empty candidate: caller treats "" as no output


class LockedRoundRobin:
    """Thread-safe provider split with failover; if every provider fails, wait and retry (rate limits)."""

    def __init__(
        self,
        clients: list[LLMClient],
        rounds: int = 20,
        pause: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.clients, self.rounds, self.pause, self._sleep = clients, rounds, pause, sleep
        self._i = 0
        self._lock = threading.Lock()

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        with self._lock:
            start = self._i
            self._i += 1
        last: Exception | None = None
        for rnd in range(self.rounds):
            for k in range(len(self.clients)):
                c = self.clients[(start + k) % len(self.clients)]
                try:
                    return c.complete(prompt, temperature=temperature, max_tokens=max_tokens)
                except (LLMError, httpx.HTTPError, ValueError) as exc:
                    last = exc
            if rnd < self.rounds - 1:
                self._sleep(self.pause)
        raise LLMError(f"all providers failed: {last}")
