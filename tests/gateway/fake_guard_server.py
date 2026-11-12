"""In-process fake of the guard Space implementing docs/CONTRACTS.md section 1 (used by the contract tests)."""

from __future__ import annotations

from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

ATTACKS = (
    ("ignore all previous instructions", "injection"),
    ("pretend you are dan", "jailbreak"),
    ("secret code", "exfiltration"),
    ("system prompt", "exfiltration"),
)


class ScoreIn(BaseModel):
    text: str = Field(min_length=1)
    kind: Literal["input", "output"] = "input"
    alias: str = "champion"


def make_fake_guard(
    models: dict[str, str | None] | None = None,
    *,
    token: str | None = None,
    legacy: bool = False,
) -> FastAPI:
    """``legacy=True`` mimics an old server that returns ``confidence`` but no ``p_unsafe``."""
    loaded = models or {"champion": "v1", "challenger": "v2", "base": "base-0"}
    app = FastAPI()

    def auth(authorization: str | None = Header(None)) -> None:
        if token is not None and authorization != f"Bearer {token}":
            raise HTTPException(401, "token required")

    @app.post("/score", dependencies=[Depends(auth)])
    def score(body: ScoreIn) -> dict[str, object]:
        alias = body.alias if loaded.get(body.alias) else "champion"  # unknown alias -> champion
        low = body.text.casefold()
        cat = next((c for needle, c in ATTACKS if needle in low), "none")
        p = 0.93 if cat != "none" else 0.04
        out: dict[str, object] = {
            "label": "UNSAFE" if p >= 0.5 else "SAFE",
            "category": cat,
            "confidence": p,
            "version": loaded[alias],
            "latency_ms": 3.2,
        }
        if not legacy:
            out["p_unsafe"] = p
        return out

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ok": True,
            "models": {k: loaded.get(k) for k in ("champion", "challenger", "base")},
        }

    return app


def bridge(app: FastAPI, base_url: str = "http://guard.test") -> httpx.Client:
    """A real ``httpx.Client`` whose transport calls ``app`` in-process.

    Starlette's TestClient now derives from ``httpx2``, whose exceptions are not ``httpx.HTTPError``; production
    code catches the real httpx errors, so tests must hand it a real httpx client.
    """
    tc = TestClient(app, base_url=base_url)

    def handler(req: httpx.Request) -> httpx.Response:
        r = tc.request(req.method, str(req.url), content=req.content, headers=dict(req.headers))
        return httpx.Response(
            r.status_code, content=r.content, headers={"content-type": r.headers["content-type"]}
        )

    return httpx.Client(transport=httpx.MockTransport(handler), base_url=base_url)
