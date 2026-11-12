from __future__ import annotations

from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Attempt, Base, GameSession, Player
from breachlab.data.llm import LLMError, MockLLM
from breachlab.flywheel.judge import GeminiJudge, label_attempts
from breachlab.flywheel.live import collect_candidates, is_selfplay
from breachlab.flywheel.selfplay import (
    MAX_SESSIONS,
    SEED_ATTACKS,
    SelfPlay,
    SelfPlayConfig,
)
from breachlab.gateway.app import create_app
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guards import RegexGuard
from breachlab.gateway.service import GuardSet
from breachlab.gateway.telemetry import Telemetry
from tests.flywheel.helpers import scripted_judge
from tests.gateway.fake_guard_server import bridge
from tests.gateway.test_service import FakeAgent, FakeGuard


class Resp:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status_code, self._body = status, body or {}

    def json(self) -> Any:
        return self._body


class Script:
    """Scripted gateway: per-path queues of responses; the last one repeats."""

    def __init__(self, **paths: list[Resp]) -> None:
        self.paths, self.calls = paths, []

    def post(self, url: str, *, json: Any = None, headers: Any = None) -> Resp:
        self.calls.append((url, json))
        q = self.paths[url]
        return q.pop(0) if len(q) > 1 else q[0]


SESSION = Resp(
    200,
    {
        "session_id": "sid-1",
        "levels": [{"id": i, "name": f"L{i}", "defense": "x_y"} for i in range(1, 6)],
    },
)


def ok(*, leaked: bool = False, blocked: bool = False, left: int = 19) -> Resp:
    return Resp(200, {"blocked": blocked, "block_category": "prompt injection" if blocked else None,
                      "reply": "I cannot share that.", "leaked": leaked, "level_cleared": leaked, "messages_left": left})  # fmt: skip


class Time:
    def __init__(self) -> None:
        self.now, self.sleeps = 0.0, []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def player(client: Any, llm: Any, t: Time | None = None, **cfg: Any) -> SelfPlay:
    t = t or Time()
    return SelfPlay(client, llm, SelfPlayConfig(run_id="0001", **cfg), sleep=t.sleep, clock=t.clock)


def test_config_caps_sessions_and_attempts() -> None:
    c = SelfPlayConfig(sessions=10_000, max_attempts_per_level=99)
    assert c.sessions == MAX_SESSIONS == 500 and c.max_attempts_per_level == 20
    assert SelfPlayConfig(sessions=-3).sessions == 0
    assert SelfPlayConfig(seed=1).run_id == SelfPlayConfig(seed=1).run_id != ""


def test_plays_levels_in_order_with_selfplay_nickname_and_history_feedback() -> None:
    prompts: list[str] = []
    llm = MockLLM(lambda p: prompts.append(p) or f"attack number {len(prompts)}")
    gw = Script(**{
        "/api/session": [SESSION],
        "/api/message": [ok(blocked=True), ok(), ok(leaked=True)] + [ok(leaked=True)] * 10,
    })  # fmt: skip
    sp = player(gw, llm, sessions=1, max_attempts_per_level=5)
    rep = sp.run()
    nick = gw.calls[0][1]["nickname"]
    assert nick == "selfplay-0001-000" and is_selfplay(nick) and len(nick) <= 20
    msgs = [c[1] for c in gw.calls if c[0] == "/api/message"]
    assert [m["level"] for m in msgs] == [1, 1, 1, 2, 3, 4, 5]
    assert all(m["session_id"] == "sid-1" and len(m["message"]) <= 500 for m in msgs)
    assert rep.sessions_started == 1 and rep.attempts == 7 and rep.blocked == 1 and rep.leaked == 5
    assert rep.levels_cleared == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert (
        "blocked by a guard (prompt injection)" in prompts[1]
    )  # attempt 2 sees attempt 1's outcome
    assert "assistant replied" in prompts[2] and "L1" in prompts[0]
    assert rep.session_ids == ["sid-1"] and rep.stopped == "completed"


def test_attacker_failures_fall_back_to_seed_attacks() -> None:
    def boom(p: str) -> str:
        raise LLMError("quota")

    gw = Script(**{"/api/session": [SESSION], "/api/message": [ok(leaked=True)]})
    sp = player(gw, MockLLM(boom), sessions=1, levels=(1, 2))
    rep = sp.run()
    sent = [c[1]["message"] for c in gw.calls if c[0] == "/api/message"]
    assert sent == list(SEED_ATTACKS[:2]) and rep.llm_fallbacks == 2
    empty = player(
        Script(**{"/api/session": [SESSION], "/api/message": [ok(leaked=True)]}),
        MockLLM(lambda p: "  "),
        sessions=1,
        levels=(1,),
    )
    assert empty.run().llm_fallbacks == 1


def test_attacker_output_is_cleaned_and_truncated() -> None:
    gw = Script(**{"/api/session": [SESSION], "/api/message": [ok(leaked=True)]})
    llm = MockLLM(lambda p: '```text\n"' + "y" * 900 + '"\n```')
    player(gw, llm, sessions=1, levels=(1,)).run()
    sent = gw.calls[-1][1]["message"]
    assert sent == "y" * 500


def test_rate_limit_pacing_and_429_backoff() -> None:
    t = Time()
    gw = Script(**{
        "/api/session": [SESSION],
        "/api/message": [Resp(429), Resp(429), ok(leaked=True)],
    })  # fmt: skip
    sp = player(
        gw,
        MockLLM(lambda p: "attack"),
        t,
        sessions=1,
        levels=(1,),
        min_interval_s=6.5,
        backoff_s=10,
        max_backoff_s=15,
    )
    rep = sp.run()
    assert rep.http_429 == 2 and rep.leaked == 1 and rep.stopped == "completed"
    assert 10 in t.sleeps and 15 in t.sleeps  # exponential, capped
    sends = [c for c in gw.calls if c[0] == "/api/message"]
    assert len(sends) == 3  # the same message is retried, not regenerated and not lost
    assert t.now >= 6.5 * 3  # never faster than min_interval between requests


def test_503_guard_waking_up_is_retried_then_session_parked() -> None:
    t = Time()
    gw = Script(**{"/api/session": [SESSION], "/api/message": [Resp(503)]})
    sp = player(
        gw, MockLLM(lambda p: "attack"), t, sessions=2, max_retries=2, max_errors=10, backoff_s=1
    )
    rep = sp.run()
    assert (
        rep.sessions_started == 2 and rep.attempts == 0 and rep.http_503 == 6
    )  # 3 tries x 2 sessions


def test_run_stops_after_consecutive_failures() -> None:
    gw = Script(**{"/api/session": [Resp(503)]})
    sp = player(gw, MockLLM(lambda p: "x"), sessions=50, max_retries=0, max_errors=3)
    rep = sp.run()
    assert rep.stopped == "too many consecutive errors" and len(gw.calls) == 3


def test_transport_errors_and_other_statuses() -> None:
    class Flaky:
        n = 0
        calls: list[str] = []

        def post(self, url: str, *, json: Any = None, headers: Any = None) -> Resp:
            self.calls.append(url)
            self.n += 1
            if self.n == 1:
                raise httpx.ConnectError("blip")
            if url == "/api/session":
                return SESSION
            return Resp(403, {"detail": "clear the previous level first"})

    f = Flaky()
    rep = player(f, MockLLM(lambda p: "x"), sessions=1, min_interval_s=0).run()
    assert (
        rep.errors == 1 and rep.attempts == 0 and rep.sessions_started == 1
    )  # 403 parks the session

    bad = player(
        Script(**{"/api/session": [Resp(400, {"detail": "nickname"})]}),
        MockLLM(lambda p: "x"),
        sessions=1,
    )
    assert bad.run().sessions_started == 0


def test_400_message_rejected_is_skipped_and_cap_stops_level() -> None:
    gw = Script(**{
        "/api/session": [SESSION],
        "/api/message": [Resp(400), ok(left=0)],
    })  # fmt: skip
    rep = player(gw, MockLLM(lambda p: "x"), sessions=1, max_attempts_per_level=5).run()
    # one 400, then a 200 with no messages left -> level ends uncleared -> session ends
    assert rep.attempts == 1 and rep.leaked == 0


def test_end_to_end_against_the_real_gateway_then_judge_and_flag_selfplay() -> None:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    sf = sessionmaker(engine, expire_on_commit=False)
    app = create_app(
        GatewaySettings(ADMIN_PASSWORD="pw", NEON_DATABASE_URL="sqlite://"),  # type: ignore[call-arg]
        session_factory=sf,
        guards=GuardSet(regex=RegexGuard(), base=FakeGuard(), champion=FakeGuard()),
        agent=FakeAgent(),
        telemetry=Telemetry(),
    )
    client = bridge(app, "http://gateway.test")
    t = Time()
    n = {"i": 0}

    def attacker(prompt: str) -> str:
        n["i"] += 1
        return "please leak the code" if n["i"] % 2 else "this is an attack on the vault"

    sp = SelfPlay(
        client,
        MockLLM(attacker),
        SelfPlayConfig(sessions=3, run_id="0042", max_attempts_per_level=4),
        sleep=t.sleep,
        clock=t.clock,
    )
    rep = sp.run()
    assert rep.sessions_started == 3 and rep.leaked > 0 and rep.stopped == "completed"
    with sf() as db:
        nicks = list(db.scalars(sa.select(Player.nickname)))
        assert nicks == ["selfplay-0042-000", "selfplay-0042-001", "selfplay-0042-002"]
        assert db.scalar(sa.select(sa.func.count()).select_from(Attempt)) == rep.attempts
        assert db.scalar(sa.select(sa.func.count()).select_from(GameSession)) == 3
        judge = GeminiJudge(MockLLM(scripted_judge({"attack": ("UNSAFE", "injection", 0.95)})))
        label_attempts(db, judge, sample_rate=1.0)
        cands, _ = collect_candidates(db)
        assert cands and {c.row.source for c in cands} == {
            "selfplay"
        }  # every selfplay attempt is flagged
