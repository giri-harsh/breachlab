import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "set_guard_url", ROOT / "scripts" / "set_guard_url.py"
)
assert spec and spec.loader
sg = importlib.util.module_from_spec(spec)
sys.modules["set_guard_url"] = sg
spec.loader.exec_module(sg)

GOOD_HEALTH = {"ok": True, "models": {"champion": "v1", "challenger": None, "base": None}}
GOOD_SCORE = {
    "label": "SAFE",
    "category": "none",
    "p_unsafe": 0.01,
    "confidence": 0.01,
    "version": "v1",
    "latency_ms": 12.0,
}


def client(handler):  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "http://guard.example.org",
        "https://placeholder.invalid",
        "https://localhost",
        "https://nodots",
        "https://10.0.0.5",
        "https://127.0.0.1",
        "https://guard.fly.dev/health",
        "https://u:p@guard.fly.dev",
        "https://guard.fly.dev?x=1",
        "ftp://guard.fly.dev",
    ],
)
def test_validate_url_rejects(raw: str) -> None:
    with pytest.raises(sg.HandoffError):
        sg.validate_url(raw)


def test_validate_url_normalises() -> None:
    assert (
        sg.validate_url(" https://breachlab-guard.fly.dev/ ") == "https://breachlab-guard.fly.dev"
    )
    assert sg.validate_url("https://8.8.8.8") == "https://8.8.8.8"


def test_health_ok_sends_bearer_token() -> None:
    seen: list[str | None] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("authorization"))
        return httpx.Response(200, json=GOOD_HEALTH)

    assert sg.check_health(client(h), "https://g.example.org", "tok")["ok"] is True
    assert seen == ["Bearer tok"]


def test_health_retries_cold_start_then_succeeds() -> None:
    calls = {"n": 0}
    waits: list[float] = []

    def h(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=GOOD_HEALTH)

    sg.check_health(client(h), "https://g.example.org", sleep=waits.append)
    assert calls["n"] == 3 and waits == [2.0, 4.0]


def test_health_gives_up_after_attempts() -> None:
    with pytest.raises(sg.HandoffError, match="never became healthy"):
        sg.check_health(
            client(lambda r: httpx.Response(502)),
            "https://g.example.org",
            attempts=2,
            sleep=lambda s: None,
        )


def test_health_network_error_is_retried_then_reported() -> None:
    def h(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(sg.HandoffError, match="ConnectError"):
        sg.check_health(client(h), "https://g.example.org", attempts=2, sleep=lambda s: None)


def test_health_auth_refusal_points_at_token() -> None:
    with pytest.raises(sg.HandoffError, match="GUARD_TOKEN"):
        sg.check_health(
            client(lambda r: httpx.Response(401)), "https://g.example.org", sleep=lambda s: None
        )


@pytest.mark.parametrize(
    "body",
    [
        {"ok": False, "models": {"champion": "v1"}},
        {"ok": True},
        {"ok": True, "models": {"champion": ""}},
        [],
    ],
)
def test_health_wrong_contract(body: object) -> None:
    with pytest.raises(sg.HandoffError, match="not"):
        sg.check_health(
            client(lambda r: httpx.Response(200, json=body)),
            "https://g.example.org",
            sleep=lambda s: None,
        )


def test_health_not_json() -> None:
    with pytest.raises(sg.HandoffError, match="JSON"):
        sg.check_health(
            client(lambda r: httpx.Response(200, text="<html>")),
            "https://g.example.org",
            sleep=lambda s: None,
        )


def test_score_ok() -> None:
    assert (
        sg.check_score(
            client(lambda r: httpx.Response(200, json=GOOD_SCORE)), "https://g.example.org"
        )["label"]
        == "SAFE"
    )


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(500),
        httpx.Response(200, text="nope"),
        httpx.Response(200, json={**GOOD_SCORE, "p_unsafe": 1.5}),
        httpx.Response(200, json={**GOOD_SCORE, "label": "MAYBE"}),
        httpx.Response(200, json={**GOOD_SCORE, "category": "x"}),
    ],
)
def test_score_rejects_bad_responses(resp: httpx.Response) -> None:
    with pytest.raises(sg.HandoffError):
        sg.check_score(client(lambda r: resp), "https://g.example.org")


def test_vercel_helpers() -> None:
    log: list[tuple[str, str]] = []

    def h(req: httpx.Request) -> httpx.Response:
        log.append((req.method, req.url.path))
        if req.url.path.endswith("/env"):
            return httpx.Response(201, json={})
        if req.method == "POST":
            return httpx.Response(200, json={"id": "dpl_1"})
        return httpx.Response(200, json={"readyState": "READY"})

    v = sg.Vercel("t", client(h))
    v.upsert_env("prj", "K", "V", sensitive=True, targets=["production"])
    assert v.redeploy("n", "prj", "main") == "dpl_1"
    v.wait_ready("dpl_1", sleep=lambda s: None)
    assert log == [
        ("POST", "/v10/projects/prj/env"),
        ("POST", "/v13/deployments"),
        ("GET", "/v13/deployments/dpl_1"),
    ]


def test_vercel_failures() -> None:
    v = sg.Vercel("t", client(lambda r: httpx.Response(403, json={})))
    with pytest.raises(sg.HandoffError, match="HTTP 403"):
        v.upsert_env("prj", "K", "V", sensitive=False, targets=["production"])
    with pytest.raises(sg.HandoffError, match="HTTP 403"):
        v.redeploy("n", "prj", "main")
    err = sg.Vercel("t", client(lambda r: httpx.Response(200, json={"readyState": "ERROR"})))
    with pytest.raises(sg.HandoffError, match="ERROR"):
        err.wait_ready("d", sleep=lambda s: None)
    with pytest.raises(sg.HandoffError, match="not READY"):
        sg.Vercel(
            "t", client(lambda r: httpx.Response(200, json={"readyState": "BUILDING"}))
        ).wait_ready("d", timeout=-1)


def test_main_rejects_placeholder_before_any_network(capsys: pytest.CaptureFixture[str]) -> None:
    assert sg.main(["--dry-run"], {"GUARD_URL": "https://placeholder.invalid"}) == 2
    assert "placeholder" in capsys.readouterr().err


def test_main_dry_run_and_full_flow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = httpx.Client

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/health":
            return httpx.Response(200, json=GOOD_HEALTH)
        if req.url.path == "/score":
            return httpx.Response(200, json=GOOD_SCORE)
        return httpx.Response(404)

    monkeypatch.setattr(
        sg.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler))
    )
    env = {"GUARD_URL": "https://breachlab-guard.fly.dev"}
    assert sg.main(["--dry-run"], env) == 0
    out = capsys.readouterr().out
    assert "champion=v1" in out and "nothing changed" in out
    assert sg.main(["--no-live"], env) == 2  # VERCEL_TOKEN missing
    assert "VERCEL_TOKEN" in capsys.readouterr().err

    calls: list[str] = []

    class FakeVercel:
        def __init__(self, token: str) -> None:
            calls.append("init")

        def upsert_env(self, pid: str, key: str, value: str, **kw: object) -> None:
            calls.append(f"env:{key}")

        def redeploy(self, name: str, pid: str, ref: str) -> str:
            calls.append(f"deploy:{name}:{ref}")
            return f"dpl_{name}"

        def wait_ready(self, dpl: str) -> None:
            calls.append(f"ready:{dpl}")

    monkeypatch.setattr(sg, "Vercel", FakeVercel)
    live: list[int] = []
    monkeypatch.setattr(sg, "run_live_spec", lambda: live.append(1) or 0)
    full = {**env, "VERCEL_TOKEN": "x", "GUARD_TOKEN": "g"}
    assert sg.main(["--ref", "main"], full) == 0
    assert calls == [
        "init",
        "env:GUARD_URL",
        "env:GUARD_TOKEN",
        "deploy:breachlab-api:main",
        "deploy:breachlab-web:main",
        "ready:dpl_breachlab-api",
        "ready:dpl_breachlab-web",
    ]
    assert live == [1]
    assert "passed" in capsys.readouterr().out
    monkeypatch.setattr(sg, "run_live_spec", lambda: 1)
    assert sg.main([], full) == 1
    assert sg.main(["--no-live"], full) == 0


def test_run_live_spec_clears_expectation_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], cwd: Path, env: dict[str, str], check: bool) -> object:
        seen.update(cmd=cmd, env=env, cwd=cwd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setenv("E2E_EXPECT_GUARD_OFFLINE", "1")
    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    assert sg.run_live_spec("https://w.example.org") == 0
    env = seen["env"]
    assert isinstance(env, dict) and "E2E_EXPECT_GUARD_OFFLINE" not in env
    assert env["E2E_LIVE"] == "1" and env["E2E_BASE_URL"] == "https://w.example.org"
    assert seen["cmd"] == ["npx", "playwright", "test", "live"]
