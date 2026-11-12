"""Ping every service in PRD 4.2 with a no-op call; print table; write reports/env-check.json."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from rich.console import Console
from rich.table import Table

TIMEOUT = 10.0


def _need(*names: str) -> str | None:
    missing = [n for n in names if not os.environ.get(n)]
    return f"missing env: {', '.join(missing)}" if missing else None


def _get(url: str, headers: dict[str, str] | None = None) -> str | None:
    try:
        r = httpx.get(url, headers=headers, timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return f"network error: {type(exc).__name__}"
    return None if r.status_code < 400 else f"HTTP {r.status_code}"


def _github() -> str | None:
    return _need("GH_TOKEN") or _get(
        "https://api.github.com/user", {"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
    )


def _hf() -> str | None:
    return _need("HF_TOKEN") or _get(
        "https://huggingface.co/api/whoami-v2",
        {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
    )


def _groq() -> str | None:
    return _need("GROQ_API_KEY") or _get(
        "https://api.groq.com/openai/v1/models",
        {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
    )


def _gemini() -> str | None:
    return _need("GEMINI_API_KEY") or _get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={os.environ['GEMINI_API_KEY']}"
    )


def _mlflow() -> str | None:
    miss = _need("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
    if miss:
        return miss
    auth = (os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"])
    try:
        r = httpx.get(
            os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
            + "/api/2.0/mlflow/experiments/search?max_results=1",
            auth=auth,
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return f"network error: {type(exc).__name__}"
    return None if r.status_code < 400 else f"HTTP {r.status_code}"


def _dagshub() -> str | None:
    return _need("DAGSHUB_TOKEN") or _get(
        "https://dagshub.com/api/v1/user",
        {"Authorization": f"Bearer {os.environ['DAGSHUB_TOKEN']}"},
    )


def _neon() -> str | None:
    """Postgres TCP first; fall back to Neon's HTTPS SQL endpoint (sandboxes allow HTTPS only)."""
    miss = _need("NEON_DATABASE_URL")
    if miss:
        return miss
    url = os.environ["NEON_DATABASE_URL"]
    try:
        import sqlalchemy as sa

        from breachlab.common.dburl import normalize_db_url

        with sa.create_engine(
            normalize_db_url(url), connect_args={"connect_timeout": 5}
        ).connect() as c:
            c.execute(sa.text("select 1"))
        return None
    except Exception:  # noqa: BLE001 - never echo the DSN
        pass
    host = urlsplit(url).hostname
    try:
        r = httpx.post(
            f"https://{host}/sql",
            headers={"Neon-Connection-String": url},
            json={"query": "select 1", "params": []},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return f"network error: {type(exc).__name__}"
    return None if r.status_code < 400 else f"HTTP {r.status_code} (https sql)"


def _langfuse() -> str | None:
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    miss = _need("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    if miss:
        return miss
    if not host:
        return "missing env: LANGFUSE_HOST (or LANGFUSE_BASE_URL)"
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    try:
        r = httpx.get(
            host.rstrip("/") + "/api/public/projects",
            auth=auth,
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return f"network error: {type(exc).__name__}"
    return None if r.status_code < 400 else f"HTTP {r.status_code}"


def _grafana() -> str | None:
    miss = _need("GRAFANA_OTLP_ENDPOINT", "GRAFANA_OTLP_INSTANCE_ID", "GRAFANA_OTLP_TOKEN")
    if miss:
        return miss
    # OTLP/HTTP basic auth: an empty metrics POST reaches auth without writing data.
    try:
        r = httpx.post(
            os.environ["GRAFANA_OTLP_ENDPOINT"].rstrip("/") + "/v1/metrics",
            auth=(os.environ["GRAFANA_OTLP_INSTANCE_ID"], os.environ["GRAFANA_OTLP_TOKEN"]),
            json={"resourceMetrics": []},
            timeout=TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return f"network error: {type(exc).__name__}"
    return None if r.status_code < 400 else f"HTTP {r.status_code}"


def _vercel() -> str | None:
    return _need("VERCEL_TOKEN") or _get(
        "https://api.vercel.com/v2/user", {"Authorization": f"Bearer {os.environ['VERCEL_TOKEN']}"}
    )


def _kaggle() -> str | None:
    miss = _need("KAGGLE_API_TOKEN")
    if miss:
        return miss
    return _get(
        "https://www.kaggle.com/api/v1/datasets/list?pageSize=1",
        {"Authorization": f"Bearer {os.environ['KAGGLE_API_TOKEN']}"},
    )


def _admin() -> str | None:
    return _need("ADMIN_PASSWORD")


CHECKS: dict[str, Callable[[], str | None]] = {
    "github": _github,
    "huggingface": _hf,
    "groq": _groq,
    "gemini": _gemini,
    "dagshub": _dagshub,
    "mlflow": _mlflow,
    "neon": _neon,
    "langfuse": _langfuse,
    "grafana": _grafana,
    "vercel": _vercel,
    "kaggle": _kaggle,
    "admin_password": _admin,
}


def run_checks(
    checks: dict[str, Callable[[], str | None]] | None = None,
) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for name, fn in (checks or CHECKS).items():
        err = fn()
        results[name] = {"status": "PASS" if err is None else "FAIL", "detail": err or "ok"}
    return results


def main() -> int:
    results = run_checks()
    table = Table(title="BreachLab env check")
    for col in ("service", "status", "detail"):
        table.add_column(col)
    for name, r in results.items():
        table.add_row(name, r["status"], r["detail"])
    Console().print(table)
    out = Path(__file__).resolve().parents[1] / "reports" / "env-check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r["status"] == "PASS" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
