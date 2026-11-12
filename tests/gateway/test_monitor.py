import numpy as np
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from breachlab.common.db import Base
from breachlab.gateway.app import create_app
from breachlab.gateway.config import GatewaySettings
from breachlab.gateway.guards import GuardOutcome, RegexGuard
from breachlab.gateway.service import GuardSet
from breachlab.monitor.canary import ATTACKS, BENIGN, PROBES, classify, run_canary
from breachlab.monitor.drift import build_report, drift_report
from tests.gateway.test_service import FakeAgent


class KeywordGuard:
    """Blocks anything that looks like one of the canary attacks."""

    def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
        bad = any(
            w in text.lower()
            for w in (
                "ignore",
                "dan",
                "backwards",
                "[system]",
                "base64",
                "verbatim",
                "batao",
                "poem",
                "rot13",
                "leetspeak",
            )
        )
        return GuardOutcome("UNSAFE" if bad else "SAFE", "injection" if bad else "none", 0.9, "kw")


def make_client(guard: object) -> tuple[TestClient, GatewaySettings]:
    engine = sa.create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    st = GatewaySettings(ADMIN_PASSWORD="pw", NEON_DATABASE_URL="sqlite://")  # type: ignore[call-arg]
    gs = GuardSet(regex=RegexGuard(), base=guard, champion=guard)  # type: ignore[arg-type]
    app = create_app(
        st,
        session_factory=sessionmaker(engine, expire_on_commit=False),
        guards=gs,
        agent=FakeAgent(),
    )
    return TestClient(app), st


def poster(client: TestClient):  # type: ignore[no-untyped-def]
    def post(url: str, body: dict, headers: dict):  # type: ignore[no-untyped-def]
        r = client.post(url.replace("http://t", ""), json=body, headers=headers)
        return r.status_code, r.json()

    return post


def test_probe_set_shape() -> None:
    assert len(PROBES) == 20 and len(BENIGN) == 10 and len(ATTACKS) == 10
    assert classify(True, 200, {"blocked": True})[0] and classify(False, 200, {"blocked": False})[0]
    assert (
        not classify(True, 200, {"blocked": False})[0]
        and not classify(False, 200, {"blocked": True})[0]
    )
    assert not classify(True, 200, {"leaked": True})[0] and not classify(True, 500, {})[0]


def test_canary_passes_with_good_guard_and_skips_leaderboard() -> None:
    client, st = make_client(KeywordGuard())
    res = run_canary("http://t", st.secret_key, poster(client), sleep=lambda _: None)
    assert len(res) == 20 and all(r.ok for r in res), [r for r in res if not r.ok]
    assert client.get("/api/leaderboard").json()["leaderboard"] == []


def test_canary_failure_increments_metric() -> None:
    class Blind:
        def score(self, text: str, kind: str = "input", alias: str = "champion") -> GuardOutcome:
            return GuardOutcome("SAFE", "none", 0.1, "blind")

    client, st = make_client(
        Blind()
    )  # a guard that passes everything -> every attack probe must fail
    events: list[str] = []

    class Tel:
        def canary_failed(self, reason: str) -> None:
            events.append(reason)

    res = run_canary(
        "http://t", st.secret_key, poster(client), telemetry=Tel(), sleep=lambda _: None
    )
    assert (
        any(not r.ok for r in res)
        and events
        and all("attack passed" in e or "blocked" in e for e in events)
    )
    # a wrong HMAC cannot start a canary session with privileges: session starts but is a normal one
    bad_status, _ = poster(client)(
        "http://t/api/message", {"session_id": "x", "level": 4, "message": "hi"}, {}
    )
    assert bad_status == 404
    res2 = run_canary(
        "http://t", "wrong-key", poster(client), telemetry=Tel(), sleep=lambda _: None
    )
    assert any(not r.ok for r in res2)


def test_canary_session_failure_reported() -> None:
    events: list[str] = []

    class Tel:
        def canary_failed(self, reason: str) -> None:
            events.append(reason)

    res = run_canary(
        "http://t", "k", lambda u, b, h: (503, {}), telemetry=Tel(), sleep=lambda _: None
    )
    assert len(res) == 1 and not res[0].ok and events == ["cannot start canary session: http 503"]


def blobs(rng: np.random.Generator, centers: list[list[float]], n: int) -> np.ndarray:
    return np.vstack([rng.normal(c, 0.15, (n, 2)) for c in centers])


def test_drift_detects_new_cluster() -> None:
    rng = np.random.default_rng(0)
    known = blobs(rng, [[0, 0], [5, 5]], 80)
    same = blobs(rng, [[0, 0], [5, 5]], 40)
    rep = drift_report(known, same)
    assert rep.share_outside_known < 0.2 and not rep.flagged
    shifted = np.vstack([blobs(rng, [[0, 0]], 40), blobs(rng, [[-6, 8]], 30)])
    rep2 = drift_report(known, shifted)
    assert rep2.share_outside_known > 0.3 and rep2.flagged and rep2.new_clusters


def test_drift_markdown() -> None:
    rng = np.random.default_rng(1)
    data = {f"k{i}": v for i, v in enumerate(blobs(rng, [[0, 0], [4, 4]], 40))}
    emb = lambda texts: np.array([data.get(t, np.array([9.0, -9.0])) for t in texts])  # noqa: E731
    rep, md = build_report(list(data), ["unseen"] * 15 + list(data)[:10], emb, "2026-W44")
    assert "Drift report 2026-W44" in md and rep.n_new == 25


def test_dashboards_and_alert_rules_are_valid_json() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "dashboards"
    for f in ("live-game", "model", "drift"):
        d = json.loads((root / f"{f}.json").read_text())
        assert d["uid"].startswith("bl-") and d["panels"]
    rules = json.loads((root / "alerts.json").read_text())["rules"]
    assert {r["uid"] for r in rules} == {
        "bl-canary",
        "bl-l4-leak",
        "bl-guard-p95",
        "bl-errors",
        "bl-spend",
    }
    metrics = (
        "requests_total",
        "errors_total",
        "guard_blocks_total",
        "leaks_total",
        "guard_latency_seconds",
        "canary_failures_total",
        "llm_cost_usd_total",
    )
    text = (root / "alerts.json").read_text()
    assert all(
        m in text
        for m in (
            "canary_failures_total",
            "leaks_total",
            "guard_latency_seconds",
            "errors_total",
            "llm_cost_usd_total",
        )
    )
    assert all(
        m
        in (root / "live-game.json").read_text()
        + (root / "model.json").read_text()
        + (root / "drift.json").read_text()
        for m in metrics[:3]
    )
