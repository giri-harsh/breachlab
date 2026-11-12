"""Observability: OTel metrics -> Grafana Cloud OTLP/HTTP (basic auth), Langfuse traces via REST ingestion.

Both sinks are optional and failure-tolerant: telemetry must never break a player's request. The vault secret
is redacted from every traced string (PRD 11).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any, Protocol

log = logging.getLogger("breachlab")

METRICS = (
    "requests_total",
    "errors_total",
    "guard_blocks_total",
    "leaks_total",
    "guard_latency_seconds",
    "request_latency_seconds",
    "guard_confidence",
    "shadow_disagreements_total",
    "canary_failures_total",
    "llm_cost_usd_total",
    "guard_unavailable_total",
    "agent_fallback_total",
    "agent_unavailable_total",
)


def redact(text: str | None, secret: str) -> str | None:
    """Replace the secret (any case, separators allowed between characters) with [SECRET]."""
    if not text or not secret:
        return text
    pat = r"[\W_]*".join(re.escape(c) for c in secret)
    return re.sub(pat, "[SECRET]", text, flags=re.IGNORECASE)


class Sink(Protocol):
    def record(self, event: dict[str, Any]) -> None: ...


def basic_auth(user: str, token: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


class Telemetry:
    """Collects per-request events and fans them out to metrics and traces."""

    def __init__(self, metrics: Any | None = None, langfuse: LangfuseClient | None = None) -> None:
        self.metrics, self.langfuse = metrics, langfuse

    def request(self, ev: dict[str, Any]) -> None:
        """``ev`` keys: session_id, level, guard_version, verdict, category, confidence, guard_latency_ms,
        agent (dict), output (dict|None), leaked, variant, latency_ms, shadow (dict|None), reply, message, secret,
        is_canary, error."""
        try:
            if self.metrics is not None:
                self.metrics.observe(ev)
            if self.langfuse is not None:
                self.langfuse.trace(ev)
        except Exception:  # noqa: BLE001
            log.exception("telemetry failure")

    def guard_unavailable(self, level: int, alias: str, kind: str = "input") -> None:
        """A guard call failed at L3-L5 and the request failed closed (503)."""
        log.error("guard unavailable: level=%s alias=%s kind=%s", level, alias, kind)
        fn = getattr(self.metrics, "guard_unavailable", None)
        if fn is not None:
            try:
                fn(level, alias, kind)
            except Exception:  # noqa: BLE001
                log.exception("telemetry failure")

    def guard_warming(self, level: int, alias: str, kind: str = "input") -> None:
        """A guard call timed out while the instance was cold: answered 202, the UI retries (ADR-0085)."""
        log.warning("guard warming: level=%s alias=%s kind=%s", level, alias, kind)
        fn = getattr(self.metrics, "guard_warming", None)
        if fn is not None:
            try:
                fn(level, alias, kind)
            except Exception:  # noqa: BLE001
                log.exception("telemetry failure")

    def _count(self, name: str, *args: Any) -> None:
        fn = getattr(self.metrics, name, None)
        if fn is not None:
            try:
                fn(*args)
            except Exception:  # noqa: BLE001
                log.exception("telemetry failure")

    def agent_fallback(self, from_model: str, to_model: str) -> None:
        """The vault agent moved to the next model in AGENT_MODELS (rate limit or error)."""
        log.warning("agent fallback: %s -> %s", from_model, to_model)
        self._count("agent_fallback", from_model, to_model)

    def agent_unavailable(self, level: int) -> None:
        """Every vault model failed inside the time budget; the player got a 503."""
        log.error("agent unavailable: level=%s", level)
        self._count("agent_unavailable", level)

    def canary_failed(self, reason: str) -> None:
        log.error("canary failure: %s", reason)
        if self.metrics is not None:
            self.metrics.canary_failure()


class OtelMetrics:  # pragma: no cover - needs the OTLP endpoint
    def __init__(
        self, endpoint: str, instance_id: str, token: str, service: str = "breachlab-gateway"
    ) -> None:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        exporter = OTLPMetricExporter(
            endpoint=endpoint.rstrip("/") + "/v1/metrics",
            headers={"Authorization": basic_auth(instance_id, token)},
        )
        self.reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
        self.provider = MeterProvider([self.reader], Resource.create({"service.name": service}))
        m = self.provider.get_meter("breachlab")
        self.requests = m.create_counter("requests_total")
        self.errors = m.create_counter("errors_total")
        self.blocks = m.create_counter("guard_blocks_total")
        self.leaks = m.create_counter("leaks_total")
        self.shadow = m.create_counter("shadow_disagreements_total")
        self.canary = m.create_counter("canary_failures_total")
        self.cost = m.create_counter("llm_cost_usd_total")
        self.unavailable = m.create_counter("guard_unavailable_total")
        self.fallbacks = m.create_counter("agent_fallback_total")
        self.agent_down = m.create_counter("agent_unavailable_total")
        self.guard_lat = m.create_histogram("guard_latency_seconds")
        self.req_lat = m.create_histogram("request_latency_seconds")
        self.conf = m.create_histogram("guard_confidence")

    def observe(self, ev: dict[str, Any]) -> None:
        a = {"level": str(ev.get("level")), "guard_version": str(ev.get("guard_version") or "none")}
        self.requests.add(1, a)
        if ev.get("error"):
            self.errors.add(1, a)
        if ev.get("verdict") == "UNSAFE":
            self.blocks.add(1, a)
        if ev.get("leaked"):
            self.leaks.add(1, a)
        if ev.get("shadow") and ev["shadow"].get("label") != ev.get("verdict"):
            self.shadow.add(1, a)
        if ev.get("cost_usd"):
            self.cost.add(float(ev["cost_usd"]), a)
        if ev.get("guard_latency_ms") is not None:
            self.guard_lat.record(float(ev["guard_latency_ms"]) / 1000, a)
        if ev.get("confidence") is not None:
            self.conf.record(float(ev["confidence"]), a)
        self.req_lat.record(float(ev.get("latency_ms", 0)) / 1000, a)
        self.reader.collect()  # serverless: push before the instance freezes

    def guard_unavailable(self, level: int, alias: str, kind: str = "input") -> None:
        self.unavailable.add(1, {"level": str(level), "alias": alias, "kind": kind})
        self.reader.collect()

    def agent_fallback(self, from_model: str, to_model: str) -> None:
        self.fallbacks.add(1, {"from_model": from_model, "to_model": to_model})
        self.reader.collect()

    def agent_unavailable(self, level: int) -> None:
        self.agent_down.add(1, {"level": str(level)})
        self.reader.collect()

    def canary_failure(self) -> None:
        self.canary.add(1, {})
        self.reader.collect()


class LangfuseClient:
    """Langfuse tracing over OpenTelemetry (the REST ingestion API is removed on 2026-11-16).

    One root span ``message`` (session id, tags, redacted input/output) with the five PRD 11 child spans.
    ``exporter`` is injectable for tests; by default an OTLP/HTTP exporter to ``{host}/api/public/otel``.
    """

    SPANS = ("input_guard", "vault_agent", "output_filter", "leak_check", "shadow_guard")

    def __init__(
        self, host: str, public_key: str, secret_key: str, exporter: Any | None = None
    ) -> None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        if exporter is None:  # pragma: no cover - network
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=host.rstrip("/") + "/api/public/otel/v1/traces",
                headers={
                    "Authorization": basic_auth(public_key, secret_key),
                    "x-langfuse-ingestion-version": "4",
                },
                timeout=5,
            )
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": "breachlab-gateway"})
        )
        self.provider.add_span_processor(SimpleSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer("breachlab")

    @staticmethod
    def span_attrs(ev: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """name -> attributes for the five child spans, secret redacted."""
        secret = str(ev.get("secret") or "")
        raw = {
            "input_guard": {"version": ev.get("guard_version"), "label": ev.get("verdict"), "category": ev.get("category"),
                            "confidence": ev.get("confidence"), "latency_ms": ev.get("guard_latency_ms")},
            "vault_agent": dict(ev.get("agent") or {"skipped": True}),
            "output_filter": dict(ev.get("output") or {"skipped": True}),
            "leak_check": {"leaked": ev.get("leaked"), "variant": ev.get("variant")},
            "shadow_guard": dict(ev.get("shadow") or {"skipped": True}),
        }  # fmt: skip
        out: dict[str, dict[str, Any]] = {}
        for name, meta in raw.items():
            out[name] = {
                f"langfuse.observation.metadata.{k}": (
                    redact(v, secret) if isinstance(v, str) else v
                )
                for k, v in meta.items()
                if v is not None
            }
        return out

    def trace(self, ev: dict[str, Any]) -> None:
        secret = str(ev.get("secret") or "")
        tags = [f"level:{ev.get('level')}"] + (["canary"] if ev.get("is_canary") else [])
        with self.tracer.start_as_current_span("message") as root:
            root.set_attribute("langfuse.session.id", str(ev.get("session_id")))
            root.set_attribute("langfuse.trace.tags", tags)
            root.set_attribute(
                "langfuse.observation.input", redact(ev.get("message"), secret) or ""
            )
            root.set_attribute("langfuse.observation.output", redact(ev.get("reply"), secret) or "")
            for name, attrs in self.span_attrs(ev).items():
                with self.tracer.start_as_current_span(name) as sp:
                    for k, v in attrs.items():
                        sp.set_attribute(k, v)


def from_env() -> Telemetry:  # pragma: no cover - env wiring
    metrics = None
    ep, iid, tok = (
        os.environ.get(k)
        for k in ("GRAFANA_OTLP_ENDPOINT", "GRAFANA_OTLP_INSTANCE_ID", "GRAFANA_OTLP_TOKEN")
    )
    if ep and iid and tok:
        metrics = OtelMetrics(ep, iid, tok)
    lf = None
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if host and pk and sk:
        lf = LangfuseClient(host, pk, sk)
    return Telemetry(metrics, lf)


def now_ms() -> float:
    return time.perf_counter() * 1000
