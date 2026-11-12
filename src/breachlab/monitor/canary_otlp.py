"""Standalone OTLP/HTTP metric push for the canary and drift jobs (Grafana Cloud, basic auth).

A short-lived CI process cannot rely on a periodic reader, so metrics are recorded into an in-memory reader,
encoded to the OTLP protobuf body and POSTed once with ``httpx``. Posting ourselves (instead of using the
SDK exporter) gives the HTTP status for the evidence trail and keeps the transport injectable for tests.
Credentials never reach logs: only the status code and a short response excerpt are reported.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("breachlab.canary")

# (url, body, headers) -> (status, response text)
OtlpPost = Callable[[str, bytes, dict[str, str]], tuple[int, str]]

CANARY_METRICS = (
    "canary_failures_total",
    "canary_ok_total",
    "canary_last_success_timestamp",
    "canary_last_run_failures",
    "canary_duration_seconds",
    "guard_cold_start_seconds",
    "guard_warm_latency_seconds",
    "guard_unavailable_total",
    "guard_up",
    "guard_ok",
    "guard_boot_seconds",
    "canary_degraded_total",
    "gateway_cold_start_seconds",
    "gateway_warm_latency_seconds",
    "breachlab_otlp_ingest_test",
)


@dataclass(frozen=True)
class PushResult:
    ok: bool
    status: int | None
    detail: str


def basic_auth(user: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


def httpx_post(
    url: str, body: bytes, headers: dict[str, str]
) -> tuple[int, str]:  # pragma: no cover - network
    import httpx

    try:
        r = httpx.post(url, content=body, headers=headers, timeout=20)
    except httpx.HTTPError as e:
        return 599, type(e).__name__
    return r.status_code, r.text[:200]


class OtlpPusher:
    """Record gauges/counters, then ``flush`` once. Retries 429/5xx with exponential backoff."""

    def __init__(
        self,
        endpoint: str,
        instance_id: str,
        token: str,
        service: str = "breachlab-canary",
        post: OtlpPost = httpx_post,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 3,
    ) -> None:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.resources import Resource

        self.url = endpoint.rstrip("/") + "/v1/metrics"
        self._auth = basic_auth(instance_id, token)
        self._post, self._sleep, self._attempts = post, sleep, attempts
        self._reader = InMemoryMetricReader()
        self._provider = MeterProvider([self._reader], Resource.create({"service.name": service}))
        self._meter = self._provider.get_meter("breachlab.canary")
        self._inst: dict[str, Any] = {}

    def gauge(self, name: str, value: float, attrs: dict[str, str] | None = None) -> None:
        g = self._inst.get(name) or self._meter.create_gauge(name)
        self._inst[name] = g
        g.set(value, attrs or {})

    def counter(self, name: str, value: float = 1, attrs: dict[str, str] | None = None) -> None:
        c = self._inst.get(name) or self._meter.create_counter(name)
        self._inst[name] = c
        c.add(value, attrs or {})

    def flush(self) -> PushResult:
        from opentelemetry.exporter.otlp.proto.common.metrics_encoder import encode_metrics

        data = self._reader.get_metrics_data()
        if data is None or not data.resource_metrics:
            return PushResult(False, None, "nothing recorded")
        body = encode_metrics(data).SerializeToString()
        headers = {"Authorization": self._auth, "Content-Type": "application/x-protobuf"}
        status, text = 0, ""
        for i in range(self._attempts):
            status, text = self._post(self.url, body, headers)
            if status < 300:
                return PushResult(True, status, "accepted")
            if status != 429 and status < 500:
                break
            if i < self._attempts - 1:
                self._sleep(2.0 * 2**i)
        return PushResult(False, status, text[:120])


def pusher_from_env(env: Mapping[str, str], **kw: Any) -> OtlpPusher | None:
    ep = env.get("GRAFANA_OTLP_ENDPOINT", "")
    iid = env.get("GRAFANA_OTLP_INSTANCE_ID", "")
    tok = env.get("GRAFANA_OTLP_TOKEN", "")
    if not (ep and iid and tok):
        return None
    return OtlpPusher(ep, iid, tok, **kw)
