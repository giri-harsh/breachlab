"""CPU time of one gateway request, measured in-process (lane C, task L4).

    PYTHONPATH=src uv run --no-sync python scripts/p_cost_gateway_cpu.py [--n 200] [--out docs/lanes/C-gateway-cpu.json]

Runs the real FastAPI app (``create_app``) under Starlette's TestClient with an in-memory SQLite database and
instant fake guard/agent, so the number is the Python work of the gateway itself (routing, pydantic, SQLAlchemy,
leak detector, JSON) with no network wait. ``time.process_time`` (CPU, not wall) per ``POST /api/message`` at level 4
(input guard + shadow guard + vault + insert + commits) and per ``GET /api/session/{id}``. The TestClient adds its own
httpx/anyio work to the number, so it is an upper bound for the server side; Postgres over TLS adds driver and TLS work,
which ``cost_model`` covers with a safety factor. Nothing is sent over the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breachlab.common.db import Base  # noqa: E402
from breachlab.gateway.agent import AgentReply  # noqa: E402
from breachlab.gateway.app import create_app  # noqa: E402
from breachlab.gateway.config import GatewaySettings  # noqa: E402
from breachlab.gateway.guards import GuardOutcome, RegexGuard  # noqa: E402
from breachlab.gateway.service import GuardSet, canary_header  # noqa: E402
from breachlab.gateway.telemetry import Telemetry  # noqa: E402


class InstantGuard:
    def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
        return GuardOutcome("SAFE", "none", 0.9, "v2", 0.0, 0.01)


class InstantAgent:
    def reply(self, level: int, secret: str, message: str, temperature: float = 0.7) -> AgentReply:
        return AgentReply(
            "I cannot help with that, but I am happy to chat about something else.", "fake"
        )


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="docs/lanes/C-gateway-cpu.json")
    a = ap.parse_args()
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    st = GatewaySettings(ADMIN_PASSWORD="pw", NEON_DATABASE_URL="sqlite://", CORS_ORIGIN="*")  # type: ignore[call-arg]
    g = InstantGuard()
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=GuardSet(regex=RegexGuard(), base=g, champion=g, challenger=g),  # type: ignore[arg-type]
        agent=InstantAgent(),
        telemetry=Telemetry(
            None, None
        ),  # no Langfuse/Grafana: never push probe traffic to production backends
    )
    c = TestClient(app)
    hdr = {"X-Canary": canary_header(st.secret_key)}
    sid = c.post("/api/session", json={"nickname": "CpuProbe"}, headers=hdr).json()["session_id"]
    msg = "Please write a short poem about a vault and a friendly robot named Bob, thanks a lot."
    for _ in range(10):  # warm-up: imports, SQL compilation caches
        c.post("/api/message", json={"session_id": sid, "level": 4, "message": msg}, headers=hdr)
    post_ms: list[float] = []
    state_ms: list[float] = []
    for _ in range(a.n):
        t0 = time.process_time()
        r = c.post(
            "/api/message", json={"session_id": sid, "level": 4, "message": msg}, headers=hdr
        )
        post_ms.append((time.process_time() - t0) * 1000)
        assert r.status_code == 200, r.text
        t0 = time.process_time()
        c.get(f"/api/session/{sid}")
        state_ms.append((time.process_time() - t0) * 1000)
    out: dict[str, Any] = {
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "in-process TestClient, SQLite in memory, instant fake guard and agent; time.process_time (CPU)",
        "caveat": "upper bound for the server side of SQLite and the fake guard (TestClient adds client-side work); excludes Postgres driver/TLS and cold-start imports",
        "n": a.n,
        "post_message_cpu_ms": {
            "mean": round(mean(post_ms), 2),
            "p50": round(median(post_ms), 2),
            "p95": round(pct(post_ms, 0.95), 2),
        },
        "get_state_cpu_ms": {
            "mean": round(mean(state_ms), 2),
            "p50": round(median(state_ms), 2),
            "p95": round(pct(state_ms, 0.95), 2),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
