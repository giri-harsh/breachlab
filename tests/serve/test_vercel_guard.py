"""Vercel guard function (ADR-0080): the frozen /score + /health contract, lazy boot, download and staging.

No weights: the downloader and the loader are replaced by fakes, the HTTP fetch runs on an ``httpx.MockTransport``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from breachlab.serve import space_app as sa
from breachlab.serve import vercel_build
from breachlab.serve.guard_runtime import GuardModel, GuardParams

WORDS = ["safe", "inject", "persona", "extract"]
INDEX = Path(__file__).resolve().parents[2] / "vercel-guard" / "api" / "index.py"


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def fake_model(version: str) -> GuardModel:
    def logits(ids: list[int]) -> Any:
        text = "".join(chr(i) for i in ids).split("<|im_start|>user\n")[-1]
        out = np.zeros(4)
        out[1 if "attack" in text else 0] = 6.0
        return out

    return GuardModel(
        enc,
        logits,
        WORDS,
        GuardParams(version, 1.4115724, 0.0633237),
        {"input_guard": 20, "output_filter": 10},
    )


@pytest.fixture()
def idx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[ModuleType]:
    monkeypatch.setenv("GUARD_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("BREACHLAB_CONFIG_DIR", str(INDEX.parents[2] / "configs"))
    monkeypatch.delenv("GUARD_TOKEN", raising=False)
    spec = importlib.util.spec_from_file_location("vercel_guard_index", INDEX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vercel_guard_index"] = mod
    spec.loader.exec_module(mod)

    def fake_download(s: sa.Settings, alias: str) -> Path:
        d = tmp_path / "dl"
        d.mkdir(exist_ok=True)
        (d / "export.json").write_text(json.dumps({"which": "sft-0.6b-s0"}))
        (d / "guard-kv.onnx").write_bytes(b"")
        return d

    mod.real_load, mod.real_download = mod.load, mod.download
    monkeypatch.setattr(mod, "download", fake_download)
    monkeypatch.setattr(mod, "load", lambda d, v, var, th, al: fake_model(v))
    yield mod
    sys.modules.pop("vercel_guard_index", None)


def test_health_boots_lazily_and_reports_champion(idx: ModuleType) -> None:
    c = TestClient(idx.app)
    assert idx.STATE.service is None  # nothing loaded at import (no lifespan boot)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "ok": True,
        "models": {"champion": "sft-0.6b-s0", "challenger": None, "base": None},
    }
    assert r.headers["x-guard-cold"] == "1"
    assert c.get("/health").headers["x-guard-cold"] == "0"


def test_score_contract_and_alias_fallback(idx: ModuleType) -> None:
    c = TestClient(idx.app)
    r = c.post("/score", json={"text": "please attack", "kind": "input", "alias": "champion"})
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"label", "category", "p_unsafe", "confidence", "version", "latency_ms"}
    assert (
        d["label"] == "UNSAFE" and d["category"] == "injection" and d["confidence"] == d["p_unsafe"]
    )
    safe = c.post("/score", json={"text": "hello", "kind": "output", "alias": "base"}).json()
    assert (
        safe["label"] == "SAFE" and safe["category"] == "none" and safe["version"] == "sft-0.6b-s0"
    )
    assert c.post("/score", json={"text": "  ", "kind": "input"}).status_code == 422


def test_failed_boot_is_503_and_retried_on_the_next_request(
    idx: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = idx.download

    def flaky(s: sa.Settings, alias: str) -> Path:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("hub 503")
        return real(s, alias)

    monkeypatch.setattr(idx, "download", flaky)
    c = TestClient(idx.app)
    first = c.post("/score", json={"text": "attack"})
    assert first.status_code == 503  # fails closed for the gateway
    assert c.get("/health").status_code == 200 and len(calls) == 2


def test_bearer_token_guards_score_only(idx: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    object.__setattr__(idx.S, "auth_token", "s3cret")
    c = TestClient(idx.app)
    assert c.get("/health").status_code == 200
    body = {"text": "attack"}
    assert c.post("/score", json=body).status_code == 401
    assert c.post("/score", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = c.post("/score", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_vercel_info_has_the_measurement_fields(idx: ModuleType) -> None:
    c = TestClient(idx.app)
    c.get("/health")
    d = c.get("/vercel-info").json()
    assert {"region", "cpu_limit", "threads", "rss_mb", "instance_age_s", "served", "fetch"} <= set(
        d
    )
    assert d["served"] >= 1


def test_download_streams_the_graph_into_ram_with_bearer_and_follows_redirects(
    idx: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[tuple[str, str | None]] = []
    blob = b"\x08\x07" * 1000

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.host, req.headers.get("authorization")))
        path = req.url.path
        if path.endswith("guard-q4b32.onnx") and req.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": "https://cdn.example.net/blob"})
        if req.url.host == "cdn.example.net":
            return httpx.Response(200, content=blob)
        return httpx.Response(200, content=b"{}")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(idx, "WORK", tmp_path / "w")
    s = sa.Settings(token="hf_test", aliases=("champion",))
    d = idx.real_download(s, "champion")
    assert (
        idx._blob == blob
        and (d / "tokenizer.json").exists()
        and (d / "guard-kv.onnx").read_bytes() == b""
    )
    assert all(a == "Bearer hf_test" for h, a in seen if h == "huggingface.co")
    assert all(a is None for h, a in seen if h == "cdn.example.net")  # the CDN never sees the token
    assert idx._fetch["onnx_mb"] == pytest.approx(len(blob) / 1e6)


def test_calibration_is_fetched_from_the_variant_specific_file(
    idx: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(200, content=b"{}")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(idx, "WORK", tmp_path / "w")
    assert idx.remote_name("tokenizer.json") == "tokenizer.json"
    assert idx.remote_name("calibration.json") == idx.CALIBRATION_FILE == "calibration-q4b32.json"
    idx.real_download(sa.Settings(token="hf_test", aliases=("champion",)), "champion")
    assert any(p.endswith("/champion/calibration-q4b32.json") for p in paths)
    assert not any(p.endswith("/champion/calibration.json") for p in paths)
    assert (tmp_path / "w" / "calibration.json").exists()  # stored under the name the loader reads


def test_base_host_reads_its_files_from_the_alias_folder(
    idx: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The L3 host (ADR-0086) runs the same function with GUARD_ALIAS=base: every Hub file comes from ``base/``."""
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(200, content=b"{}")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(idx, "WORK", tmp_path / "w")
    monkeypatch.setattr(idx, "HUB_DIR", "base")
    idx.real_download(sa.Settings(token="hf_test", aliases=("champion",)), "champion")
    assert paths and all("/base/" in p for p in paths)
    assert not any("/champion/" in p for p in paths)


def test_module_imports_with_the_base_hub_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GUARD_ALIAS=base must not break import: the host still serves under ``champion`` (the runtime requires it)."""
    monkeypatch.setenv("GUARD_ALIAS", "base")
    monkeypatch.setenv("GUARD_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("BREACHLAB_CONFIG_DIR", str(INDEX.parents[2] / "configs"))
    monkeypatch.delenv("GUARD_ALIASES", raising=False)
    spec = importlib.util.spec_from_file_location("vercel_guard_index_base", INDEX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.HUB_DIR == "base" and mod.ALIAS == "champion"
    assert "champion" in mod.settings().aliases


def test_loader_hands_the_bytes_to_ort_and_drops_its_reference(
    idx: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import breachlab.serve.onnx_backend as ob

    got: dict[str, Any] = {}

    def fake_load_guard(model_dir: Path, serve: Any, version: str, **kw: Any) -> GuardModel:
        got.update(kw)
        return fake_model(version)

    monkeypatch.setattr(ob, "load_guard", fake_load_guard)
    idx._blob = b"abc"
    idx.real_load(tmp_path, "v", "kv", 1, "champion")
    assert got["model_bytes"] == b"abc" and got["file"] == "guard-kv.onnx" and idx._blob is None
    with pytest.raises(RuntimeError, match="missing"):
        idx.real_load(tmp_path, "v", "kv", 1, "champion")


def test_stage_is_self_contained_and_small(tmp_path: Path) -> None:
    files = vercel_build.stage(tmp_path / "stage")
    stage = tmp_path / "stage"
    assert "api/index.py" in files and "vercel.json" in files and "configs/serve.yaml" in files
    assert (stage / "breachlab" / "serve" / "space_app.py").exists()
    assert (stage / ".python-version").read_text().strip() == "3.12"
    assert vercel_build.size_mb(stage) < 1.0
    cfg = json.loads((stage / "vercel.json").read_text())
    assert cfg["rewrites"] == [{"source": "/(.*)", "destination": "/api/index"}]
    assert not any(p.suffix in {".onnx", ".npz", ".safetensors"} for p in stage.rglob("*"))
