"""Deployment-facing behaviour of the guard service: CPU sizing, aliases, variant fallback, auth, boot retry."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from breachlab.serve import space_app
from breachlab.serve.guard_runtime import GuardModel
from breachlab.serve.space_app import Settings, create_app
from tests.serve.test_space_app import fake_loader, make_dirs


def test_cpu_limit_honours_cgroup_quota(tmp_path: Path) -> None:
    cap = len(os.sched_getaffinity(0))
    assert space_app.cpu_limit(tmp_path) == cap  # no cgroup files: affinity only
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert space_app.cpu_limit(tmp_path) == cap
    (tmp_path / "cpu.max").write_text("100000 100000\n")
    assert space_app.cpu_limit(tmp_path) == 1
    (tmp_path / "cpu.max").write_text("150000 100000\n")  # 1.5 CPUs rounds up
    assert space_app.cpu_limit(tmp_path) == min(2, cap)
    (tmp_path / "cpu.max").write_text("garbage")
    assert space_app.cpu_limit(tmp_path) == cap


def test_cpu_limit_cgroup_v1(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("100000\n")
    (tmp_path / "cpu/cpu.cfs_period_us").write_text("100000\n")
    assert space_app.cpu_limit(tmp_path) == 1
    (tmp_path / "cpu/cpu.cfs_quota_us").write_text("-1\n")  # unlimited
    assert space_app.cpu_limit(tmp_path) >= 1


def test_settings_aliases_token_and_sync_from_env() -> None:
    s = Settings.from_env(
        {"GUARD_ALIASES": "champion, base", "GUARD_TOKEN": "t0k", "GUARD_BOOT_SYNC": "1"}
    )
    assert s.aliases == ("champion", "base") and s.boot_sync and s.auth_token == "t0k"
    assert "t0k" not in repr(s)
    d = Settings.from_env({})
    assert d.aliases == space_app.ALIASES and not d.boot_sync and d.auth_token is None
    for bad in ("base", "champion,nope"):
        with pytest.raises(ValueError):
            Settings.from_env({"GUARD_ALIASES": bad})


def test_only_requested_aliases_are_loaded(tmp_path: Path) -> None:
    s = Settings(model_dir=make_dirs(tmp_path), aliases=("champion", "base"))
    app = create_app(s, loader=fake_loader, background=False)
    with TestClient(app) as c:
        assert c.get("/health").json()["models"] == {
            "champion": "w-champion",
            "challenger": None,
            "base": "w-base",
        }
        assert c.get("/info").json()["variants"] == {"champion": "fp32", "base": "fp32"}


def test_bearer_token_guards_score_but_not_health(tmp_path: Path) -> None:
    s = Settings(model_dir=make_dirs(tmp_path), auth_token="s3cret")
    app = create_app(s, loader=fake_loader, background=False)
    body = {"text": "attack"}
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/info").status_code == 200
        assert c.post("/score", json=body).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert c.post("/score", json=body, headers=bad).status_code == 401
        good = {"Authorization": "Bearer s3cret"}
        assert c.post("/score", json=body, headers=good).status_code == 200


def test_effective_variant_falls_back_to_fp32(tmp_path: Path) -> None:
    assert space_app.effective_variant(tmp_path, "kv") == "kv"  # nothing there: loader reports it
    (tmp_path / "guard-fp32.onnx").write_text("x")
    assert space_app.effective_variant(tmp_path, "kv") == "fp32"
    (tmp_path / "guard-kv.onnx").write_text("x")
    assert space_app.effective_variant(tmp_path, "kv") == "kv"
    assert space_app.effective_variant(tmp_path, "fp32") == "fp32"


def test_boot_uses_per_alias_variant(tmp_path: Path) -> None:
    root = make_dirs(tmp_path)
    for a in ("champion", "challenger", "base"):
        (root / a / "guard-fp32.onnx").write_text("x")
    (root / "champion" / "guard-kv.onnx").write_text("x")
    seen: dict[str, str] = {}

    def spy(directory: Path, version: str, variant: str, threads: int, alias: str) -> GuardModel:
        seen[alias] = variant
        return fake_loader(directory, version, variant, threads, alias)

    app = create_app(Settings(model_dir=root, variant="kv"), loader=spy, background=False)
    with TestClient(app) as c:
        assert c.get("/info").json()["variants"] == seen
    assert seen == {"champion": "kv", "challenger": "fp32", "base": "fp32"}


def test_boot_with_retry_recovers_from_transient_failure(tmp_path: Path) -> None:
    calls: list[str] = []
    root = make_dirs(tmp_path)

    def flaky_download(s: Settings, alias: str) -> Path:
        calls.append(alias)
        if len(calls) == 1:
            raise RuntimeError("503 from the hub")
        return root / alias

    state = space_app.State()
    waits: list[float] = []
    space_app.boot_with_retry(
        Settings(), state, flaky_download, fake_loader, wait_s=(1.0,), sleep=waits.append
    )
    assert state.service is not None and state.boot.error is None and waits == [1.0]


def test_boot_with_retry_gives_up_and_keeps_the_error() -> None:
    def boom(s: Settings, alias: str) -> Path:
        raise RuntimeError("401")

    state = space_app.State()
    waits: list[float] = []
    space_app.boot_with_retry(
        Settings(), state, boom, fake_loader, attempts=3, wait_s=(1.0, 2.0), sleep=waits.append
    )
    assert state.service is None and "RuntimeError" in (state.boot.error or "")
    assert waits == [1.0, 2.0]


def test_boot_sync_setting_blocks_until_loaded(tmp_path: Path) -> None:
    s = Settings(model_dir=make_dirs(tmp_path), boot_sync=True)
    app = create_app(s, loader=fake_loader, background=True)  # sync despite background=True
    with TestClient(app) as c:
        assert app.state.guard.service is not None
        assert c.get("/health").status_code == 200
