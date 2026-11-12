import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from breachlab.serve import space_app
from breachlab.serve.guard_runtime import GuardModel, GuardParams
from breachlab.serve.space_app import Settings, create_app

WORDS = ["safe", "inject", "persona", "extract"]
MAXTOK = {"input_guard": 20, "output_filter": 10}


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def fake_model(version: str, hot: int, temperature: float = 1.0, thr: float = 0.5) -> GuardModel:
    """Logits are driven by the message: the 'hot' class wins when the text contains 'attack'."""

    def logits(ids: list[int]) -> Any:
        text = "".join(chr(i) for i in ids)
        out = np.zeros(4)
        out[hot if "attack" in text.split("<|im_start|>user\n")[-1] else 0] = 6.0
        return out

    return GuardModel(enc, logits, WORDS, GuardParams(version, temperature, thr), MAXTOK)


def make_dirs(
    tmp_path: Path, aliases: tuple[str, ...] = ("champion", "challenger", "base")
) -> Path:
    for a in aliases:
        d = tmp_path / a
        d.mkdir(parents=True)
        (d / "export.json").write_text(json.dumps({"which": f"w-{a}"}))
    return tmp_path


def fake_loader(
    directory: Path, version: str, variant: str, threads: int, alias: str
) -> GuardModel:
    hot = {"champion": 1, "challenger": 2, "base": 3}[alias]
    return fake_model(version, hot)


def client(tmp_path: Path, **kw: Any) -> TestClient:
    s = Settings(
        model_dir=make_dirs(tmp_path, kw.pop("aliases", ("champion", "challenger", "base")))
    )
    app = create_app(s, loader=kw.pop("loader", fake_loader), background=False)
    return TestClient(app)


def test_score_contract(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        r = c.post("/score", json={"text": "please attack", "kind": "input", "alias": "champion"})
        assert r.status_code == 200
        d = r.json()
        assert set(d) == {"label", "category", "p_unsafe", "confidence", "version", "latency_ms"}
        assert d["label"] == "UNSAFE" and d["category"] == "injection"
        assert d["confidence"] == d["p_unsafe"] and 0.5 < d["p_unsafe"] <= 1
        assert d["version"] == "w-champion" and d["latency_ms"] >= 0
        safe = c.post("/score", json={"text": "hello", "kind": "output", "alias": "base"}).json()
        assert (
            safe["label"] == "SAFE" and safe["category"] == "none" and safe["version"] == "w-base"
        )
        ch = c.post("/score", json={"text": "attack", "alias": "challenger"}).json()
        assert ch["category"] == "jailbreak"
        assert c.post("/score", json={"text": "attack", "alias": "base"}).json()["category"] == (
            "exfiltration"
        )


def test_unknown_alias_falls_back_to_champion_and_defaults(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        d = c.post("/score", json={"text": "attack", "alias": "nope"}).json()
        assert d["version"] == "w-champion"
        assert c.post("/score", json={"text": "attack"}).json()["version"] == "w-champion"


def test_validation_errors(tmp_path: Path) -> None:
    with client(tmp_path) as c:
        assert c.post("/score", json={"text": "x", "kind": "bogus"}).status_code == 422
        assert c.post("/score", json={"kind": "input"}).status_code == 422
        too_long = "a" * (space_app.MAX_CHARS + 1)
        assert c.post("/score", json={"text": too_long}).status_code == 422


def test_health_lists_models_and_nulls(tmp_path: Path) -> None:
    with client(tmp_path, aliases=("champion",)) as c:
        h = c.get("/health")
        assert h.status_code == 200
        assert h.json() == {
            "ok": True,
            "models": {"champion": "w-champion", "challenger": None, "base": None},
        }
        info = c.get("/info").json()
        assert info["ready"] is True and info["variant"] == "fp32"
        assert info["cold_start_s"] is not None and "champion" in info["load_s"]


def test_missing_optional_alias_downloads_fail_softly(tmp_path: Path) -> None:
    def download(s: Settings, alias: str) -> Path:
        if alias != "champion":
            raise FileNotFoundError(alias)
        return make_dirs(tmp_path, ("champion",)) / "champion"

    app = create_app(Settings(), download=download, loader=fake_loader, background=False)
    with TestClient(app) as c:
        assert c.get("/health").json()["models"]["base"] is None
        assert (
            c.post("/score", json={"text": "hi", "alias": "base"}).json()["version"] == "w-champion"
        )


def test_boot_failure_fails_closed(tmp_path: Path) -> None:
    def boom(s: Settings, alias: str) -> Path:
        raise RuntimeError("401 unauthorized")

    app = create_app(Settings(), download=boom, loader=fake_loader, background=False)
    with TestClient(app) as c:
        h = c.get("/health")
        assert h.status_code == 503 and h.json()["ok"] is False
        assert "RuntimeError" in h.json()["error"]
        assert c.post("/score", json={"text": "hi"}).status_code == 503
        assert c.get("/info").json()["ready"] is False


def test_inference_error_is_503_not_safe(tmp_path: Path) -> None:
    def bad_loader(
        directory: Path, version: str, variant: str, threads: int, alias: str
    ) -> GuardModel:
        def logits(ids: list[int]) -> Any:
            raise RuntimeError("ort failed")

        return GuardModel(enc, logits, WORDS, GuardParams(version), MAXTOK)

    with client(tmp_path, loader=bad_loader) as c:
        assert c.post("/score", json={"text": "hi"}).status_code == 503


def test_background_boot_reports_503_until_ready(tmp_path: Path) -> None:
    gate = threading.Event()

    def slow_loader(
        directory: Path, version: str, variant: str, threads: int, alias: str
    ) -> GuardModel:
        gate.wait(5)
        return fake_loader(directory, version, variant, threads, alias)

    s = Settings(model_dir=make_dirs(tmp_path))
    app = create_app(s, loader=slow_loader, background=True)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/score", json={"text": "hi"}).status_code == 503
        gate.set()
        state = app.state.guard
        for _ in range(100):
            if state.service is not None:
                break
            threading.Event().wait(0.05)
        assert c.get("/health").status_code == 200


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "GUARD_VARIANT": "w8",
            "GUARD_THREADS": "3",
            "GUARD_MODEL_DIR": "/x",
            "HF_TOKEN": "tok-123",
        }
    )
    assert (
        s.variant == "w8" and s.threads == 3 and s.model_dir == Path("/x") and s.token == "tok-123"
    )
    assert "tok-123" not in repr(s)  # the token is never repr'd
    assert Settings.from_env({}).variant == "fp32"
    with pytest.raises(ValueError):
        Settings.from_env({"GUARD_VARIANT": "int4"})


def test_broken_optional_alias_is_skipped_but_broken_champion_is_fatal(tmp_path: Path) -> None:
    def flaky(directory: Path, version: str, variant: str, threads: int, alias: str) -> GuardModel:
        if alias == "challenger":
            raise RuntimeError("missing guard-kv.onnx")
        return fake_loader(directory, version, variant, threads, alias)

    with client(tmp_path, loader=flaky) as c:
        models = c.get("/health").json()["models"]
        assert models["challenger"] is None and models["champion"] == "w-champion"

    def no_champion(
        directory: Path, version: str, variant: str, threads: int, alias: str
    ) -> GuardModel:
        raise RuntimeError("bad file")

    with client(tmp_path / "x", loader=no_champion) as c:
        assert c.get("/health").status_code == 503


def test_max_tokens_override_applies_to_every_model(tmp_path: Path) -> None:
    s = Settings.from_env({"GUARD_MAX_TOKENS": "7", "GUARD_MODEL_DIR": str(make_dirs(tmp_path))})
    assert s.max_tokens == 7
    app = create_app(s, loader=fake_loader, background=False)
    with TestClient(app):
        svc = app.state.guard.service
        assert all(
            m.max_tokens == {"input_guard": 7, "output_filter": 7} for m in svc.models.values()
        )
    assert Settings.from_env({}).max_tokens is None


def test_version_of_falls_back_to_alias(tmp_path: Path) -> None:
    assert space_app.version_of(tmp_path, "base") == "base"
    (tmp_path / "export.json").write_text("{}")
    assert space_app.version_of(tmp_path, "base") == "base"
