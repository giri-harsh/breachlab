"""FastAPI app (Modal ASGI / Vercel / Docker). All routes are thin wrappers over GameService."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator
from typing import Any

import sqlalchemy as sa
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from breachlab.common.db import Attempt, Base, GameSession, Label
from breachlab.common.dburl import normalize_db_url
from breachlab.gateway import extras
from breachlab.gateway.agent import GroqAgent, VaultAgent
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guard_health import GuardProbe
from breachlab.gateway.guards import HttpGuard, RegexGuard
from breachlab.gateway.levels import load_levels
from breachlab.gateway.ratelimit import SlidingWindow
from breachlab.gateway.service import (
    DISPLAY_REVIEWER,
    GameError,
    GameService,
    GuardSet,
    canary_valid,
)
from breachlab.gateway.serving import ServingState, set_default, source_from_env
from breachlab.gateway.telemetry import Telemetry, from_env


class StartIn(BaseModel):
    nickname: str = Field(min_length=1, max_length=40)


class MessageIn(BaseModel):
    session_id: str
    level: int = Field(ge=1, le=6)  # 6 = the daily boss (CONTRACTS 7.7)
    message: str = Field(min_length=1, max_length=2000)


class HintIn(BaseModel):
    level: int = Field(ge=1, le=5)
    tier: int = Field(ge=1, le=3)


class SkipIn(BaseModel):
    level: int = Field(ge=1, le=5)


class LabelIn(BaseModel):
    attempt_id: int
    label: str = Field(pattern="^(SAFE|UNSAFE)$")
    category: str | None = None
    display: bool = False  # approve showing this winning prompt to players who skip (CONTRACTS 7.5)


WARM_RETRY_S = 5  # Retry-After for a 202 "warming" answer (ADR-0085)


def create_app(
    settings: GatewaySettings | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
    guards: GuardSet | None = None,
    agent: VaultAgent | None = None,
    telemetry: Telemetry | None = None,
    guard_probe: GuardProbe | None = None,
) -> FastAPI:
    st = settings or GatewaySettings()
    if session_factory is None:
        engine = sa.create_engine(normalize_db_url(st.database_url), pool_pre_ping=True)
        session_factory = sessionmaker(engine, expire_on_commit=False)
    if guards is None:

        def host(url: str) -> HttpGuard | None:
            return (
                HttpGuard(
                    url,
                    st.guard_timeout_s,
                    token=st.guard_token,
                    cold_timeout=st.guard_cold_timeout_s,
                    cold_after_s=st.guard_cold_after_s,
                    warm_grace_s=st.guard_warm_grace_s,
                )
                if url
                else None
            )

        http = host(st.guard_url)
        base_http = (
            host(st.guard_base_url) or http
        )  # L3 has its own host when GUARD_BASE_URL is set (ADR-0086)
        serving = ServingState(source_from_env(), ttl=st.champion_ttl_s)
        set_default(serving)
        guards = GuardSet(
            regex=RegexGuard(), base=base_http, champion=http, challenger=http, serving=serving
        )
    tel = telemetry if telemetry is not None else from_env()
    the_agent = agent or GroqAgent(st.groq_api_key, st.vault_models, on_fallback=tel.agent_fallback)
    app = FastAPI(title="BreachLab gateway", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in st.cors_origin.split(",")],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def db() -> Iterator[Session]:
        with session_factory() as s:
            yield s

    def svc(s: Session = Depends(db)) -> GameService:
        return GameService(s, guards, the_agent, st.secret_key, st.salt, telemetry=tel)

    def client_ip(req: Request) -> str:
        fwd = req.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (req.client.host if req.client else "0.0.0.0")  # noqa: S104

    def wrap(fn: Any) -> Any:
        try:
            return fn()
        except GameError as e:
            if (
                e.status == 202
            ):  # the guard is booting (ADR-0085): not an error, the UI waits and retries
                return JSONResponse(
                    {"status": "warming", "retry_after_s": WARM_RETRY_S, "detail": e.detail},
                    status_code=202,
                    headers={"Retry-After": str(WARM_RETRY_S)},
                )
            raise HTTPException(e.status, e.detail) from None

    probe = guard_probe or GuardProbe(st.guard_url, token=st.guard_token)
    champion_http = getattr(guards, "champion", None)
    if isinstance(champion_http, HttpGuard) and hasattr(probe, "attach"):
        probe.attach(champion_http)  # message-path and /api/warm evidence count for guard_state
    lim = load_levels().limits
    hint_limit = SlidingWindow(lim.hint_rate_per_session_per_min)
    skip_limit = SlidingWindow(lim.skip_rate_per_session_per_min)
    admin_limit = SlidingWindow(lim.admin_rate_per_ip_per_min)

    app.include_router(
        extras.build_router(db, st)
    )  # v1.1 daily board and stats (lane V, CONTRACTS 7.7-7.8)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        # ``guard`` = configured (compat).
        # ``guard_state`` = ok | warming | down (a cold guard is "warming", never "down"); ``guard_ok`` is true only for "ok".
        state = probe.state()
        out: dict[str, Any] = {
            "ok": True,
            "guard": bool(st.guard_url),
            "guard_ok": state == "ok",
            "guard_state": state,
        }
        if guards.serving is not None:
            out["serving_version"] = guards.serving.version()
        return out

    @app.get("/api/warm")
    def warm() -> Any:
        """Wake the guard host(s) before the player types (ADR-0085). The web app fires this without waiting.

        Pings each distinct HTTP guard's ``/health`` (rate limited per guard). 200 when all are warm, 202 while any is
        still booting, 200 with ``down`` when one failed after its grace window; it never reveals hosts or tokens.
        """
        seen: dict[int, tuple[str, HttpGuard]] = {}
        for name in ("champion", "base"):
            g = getattr(guards, name, None)
            if isinstance(g, HttpGuard):
                seen.setdefault(id(g), (name, g))
        results = {name: g.warm() for name, g in seen.values()}
        states = {r["status"] for r in results.values()}
        status = "down" if "down" in states else "warming" if "warming" in states else "warm"
        if not results:
            status = "none"
        return JSONResponse(
            {"status": status, "guards": results},
            status_code=202 if status == "warming" else 200,
        )

    @app.post("/api/session")
    def start(
        body: StartIn,
        req: Request,
        g: GameService = Depends(svc),
        x_canary: str | None = Header(None),
    ) -> Any:
        return wrap(
            lambda: g.start_session(
                body.nickname, client_ip(req), canary_valid(st.secret_key, x_canary)
            )
        )

    @app.get("/api/session/{sid}")
    def state(sid: str, g: GameService = Depends(svc)) -> Any:
        return wrap(lambda: g.state(sid))

    @app.post("/api/message")
    def message(body: MessageIn, req: Request, g: GameService = Depends(svc)) -> Any:
        return wrap(
            lambda: g.send(body.session_id, body.level, body.message, g.ip_hash(client_ip(req)))
        )

    @app.post("/api/session/{sid}/hint")
    def hint(sid: str, body: HintIn, g: GameService = Depends(svc)) -> Any:
        if not hint_limit.hit(sid):
            raise HTTPException(429, "slow down: 10 hint requests per minute per session")
        return wrap(lambda: g.hint(sid, body.level, body.tier))

    @app.post("/api/session/{sid}/skip")
    def skip(sid: str, body: SkipIn, g: GameService = Depends(svc)) -> Any:
        if not skip_limit.hit(sid):
            raise HTTPException(429, "slow down: 10 skip requests per minute per session")
        return wrap(lambda: g.skip(sid, body.level))

    @app.get("/api/leaderboard")
    def board(g: GameService = Depends(svc)) -> Any:
        return {"leaderboard": g.leaderboard(), "hall_of_breaches": g.hall_of_breaches()}

    def admin(req: Request, authorization: str | None = Header(None)) -> None:
        # PRD 12: every admin request counts, failed passwords included, so guessing is capped at 5 a minute per IP.
        ip_key = hashlib.sha256(f"{st.salt}|{client_ip(req)}".encode()).hexdigest()
        if not admin_limit.hit(ip_key):
            raise HTTPException(429, "too many admin requests, wait a minute")
        pw = (authorization or "").removeprefix("Bearer ")
        if not st.admin_password or not hmac.compare_digest(pw, st.admin_password):
            raise HTTPException(401, "admin password required")

    @app.get("/api/admin/queue", dependencies=[Depends(admin)])
    def queue(limit: int = 20, s: Session = Depends(db)) -> Any:
        """Unreviewed attempts the judge/guard disagree on or are low-confidence (review queue, PRD 10)."""
        q = (
            sa.select(Attempt)
            .outerjoin(Label, Label.attempt_id == Attempt.id)
            .join(GameSession, GameSession.id == Attempt.session_id)
            .where(
                GameSession.is_canary.is_(False),
                sa.or_(Label.attempt_id.is_(None), Label.source != "human"),
            )
            .order_by(sa.func.abs(sa.func.coalesce(Attempt.confidence, 0.5) - 0.5))
            .limit(limit)
        )
        return [
            {"id": a.id, "message": a.message, "verdict": a.verdict, "category": a.category,
             "confidence": a.confidence, "leaked": a.leaked, "level": a.level}
            for a in s.scalars(q)
        ]  # fmt: skip

    @app.post("/api/admin/label", dependencies=[Depends(admin)])
    def label(body: LabelIn, s: Session = Depends(db)) -> Any:
        existing = s.get(Label, body.attempt_id)
        reviewer = DISPLAY_REVIEWER if body.display else "admin"
        if existing is None:
            s.add(
                Label(
                    attempt_id=body.attempt_id,
                    label=body.label,
                    category=body.category,
                    source="human",
                    reviewer=reviewer,
                )
            )
        else:
            existing.label, existing.category, existing.source, existing.reviewer = (
                body.label,
                body.category,
                "human",
                reviewer,
            )
        s.commit()
        return {"ok": True}

    if st.database_url.startswith("sqlite"):
        Base.metadata.create_all(session_factory.kw["bind"])
    return app
