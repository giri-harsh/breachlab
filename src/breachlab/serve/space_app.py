"""Guard service, host-agnostic (docs/CONTRACTS.md section 1; hosting options in docs/deploy-guard.md).

``POST /score`` and ``GET /health`` implement the frozen contract. Models (champion / challenger / base) are
pulled from the HF model repo at boot in a background thread, so the port answers (503 on /health) while they
load. Inference is serialized behind a lock: a 2-vCPU host is already saturated by one ORT session, so concurrent
sessions would only fight for cores. ``GET /info`` (not part of the contract) reports the variant, ORT version
and the measured cold-start timings. The same image runs on a Hugging Face Docker Space, Cloud Run, Fly.io or a
plain VM; only the environment differs (``Settings.from_env``).
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from breachlab.serve.guard_runtime import GuardModel, GuardService

log = logging.getLogger("guard-space")

ALIASES = ("champion", "challenger", "base")
MAX_CHARS = 20_000

Variant = Literal["fp32", "kv", "w8", "dyn_int8", "dyn_pc"]
VARIANT_FILES: dict[str, str] = {
    "fp32": "guard-fp32.onnx",
    "kv": "guard-kv.onnx",
    "w8": "guard-w8.onnx",
    "dyn_int8": "guard-int8.onnx",
    "dyn_pc": "guard-dynpc.onnx",
}


def cpu_limit(cgroup: Path = Path("/sys/fs/cgroup")) -> int:
    """CPUs this container may really use: affinity mask, tightened by the cgroup CPU quota (v2, then v1).

    ``os.cpu_count()`` reports the host's cores on Cloud Run / Fly / Docker with ``--cpus``; sizing the ONNX
    Runtime thread pool from it oversubscribes the quota and wrecks the tail latency.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    try:
        quota, period = (cgroup / "cpu.max").read_text(encoding="utf-8").split()[:2]
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        try:
            q = int((cgroup / "cpu/cpu.cfs_quota_us").read_text(encoding="utf-8"))
            per = int((cgroup / "cpu/cpu.cfs_period_us").read_text(encoding="utf-8"))
            if q > 0 and per > 0:
                n = min(n, math.ceil(q / per))
        except (OSError, ValueError):
            pass
    return max(1, n)


class ScoreRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    kind: Literal["input", "output"] = "input"
    alias: str = "champion"

    @model_validator(mode="after")
    def _input_not_empty(self) -> ScoreRequest:
        """An empty user message is a client error; an empty assistant reply is legitimate (scored as is)."""
        if self.kind == "input" and not self.text.strip():
            raise ValueError("text must not be empty for kind=input")
        return self


class ScoreResponse(BaseModel):
    label: Literal["SAFE", "UNSAFE"]
    category: Literal["none", "injection", "jailbreak", "exfiltration"]
    p_unsafe: float
    confidence: float  # equals p_unsafe (kept for the gateway's HttpGuard)
    version: str
    latency_ms: float
    stage: str | None = (
        None  # cascade guard only (student | teacher | failed_closed); absent otherwise
    )


@dataclass(frozen=True)
class Settings:
    repo: str = "Pseudo-Harsh/breachlab-guard-v1"
    variant: str = "fp32"
    threads: int = 2
    model_dir: Path | None = (
        None  # local directory holding champion/ challenger/ base/ (skips the download)
    )
    cache_dir: Path = Path("/tmp/hf-guard")
    token: str | None = field(default=None, repr=False)
    max_tokens: int | None = (
        None  # override of the message token limit (ADR-0039 rung 4); None = configs/serve.yaml
    )
    aliases: tuple[str, ...] = ALIASES  # which aliases to load; champion is mandatory
    auth_token: str | None = field(default=None, repr=False)  # bearer token required on /score
    boot_sync: bool = False  # load models before the port opens (hosts that throttle idle CPU)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)
        variant = e.get("GUARD_VARIANT", "fp32")
        if variant not in VARIANT_FILES:
            raise ValueError(
                f"GUARD_VARIANT must be one of {sorted(VARIANT_FILES)}, got {variant!r}"
            )
        md = e.get("GUARD_MODEL_DIR")
        aliases = tuple(
            a.strip() for a in e.get("GUARD_ALIASES", ",".join(ALIASES)).split(",") if a.strip()
        )
        if "champion" not in aliases or not set(aliases) <= set(ALIASES):
            raise ValueError(
                f"GUARD_ALIASES must include champion and only {ALIASES}, got {aliases}"
            )
        return cls(
            repo=e.get("GUARD_MODEL_REPO", cls.repo),
            variant=variant,
            threads=int(e.get("GUARD_THREADS") or cpu_limit()),
            model_dir=Path(md) if md else None,
            cache_dir=Path(e.get("GUARD_CACHE_DIR", "/tmp/hf-guard")),
            max_tokens=int(e["GUARD_MAX_TOKENS"]) if e.get("GUARD_MAX_TOKENS") else None,
            token=e.get("HF_TOKEN") or None,
            aliases=aliases,
            auth_token=e.get("GUARD_TOKEN") or None,
            boot_sync=e.get("GUARD_BOOT_SYNC", "") in {"1", "true", "yes"},
        )


@dataclass
class Boot:
    """Cold-start record: seconds spent downloading and loading each alias."""

    started: float = field(default_factory=time.perf_counter)
    download_s: dict[str, float] = field(default_factory=dict)
    load_s: dict[str, float] = field(default_factory=dict)
    total_s: float | None = None
    error: str | None = None


class State:
    def __init__(self) -> None:
        self.service: GuardService | None = None
        self.versions: dict[str, str | None] = {a: None for a in ALIASES}
        self.boot = Boot()
        self.variant = "fp32"
        self.variants: dict[
            str, str
        ] = {}  # alias -> variant actually loaded (fp32 if the requested is absent)
        self.lock = threading.Lock()


Download = Callable[[Settings, str], Path]
Loader = Callable[[Path, str, str, int, str], GuardModel]


def download_alias(s: Settings, alias: str) -> Path:  # pragma: no cover - network
    """Fetch only the files of one alias and the chosen variant (not all variants) from the model repo."""
    from huggingface_hub import HfApi, snapshot_download

    f = VARIANT_FILES[s.variant]
    if s.variant != "fp32" and not HfApi(token=s.token).file_exists(
        s.repo, f"{alias}/{f}", token=s.token
    ):
        log.warning("%s has no %s; falling back to fp32", alias, s.variant)
        f = VARIANT_FILES["fp32"]
    root = snapshot_download(
        s.repo,
        token=s.token,
        cache_dir=str(s.cache_dir),
        allow_patterns=[
            f"{alias}/{f}",
            f"{alias}/{f}.data",
            f"{alias}/prefix_kv_*.npz",
            f"{alias}/tokenizer.json",
            f"{alias}/calibration.json",
            f"{alias}/export.json",
        ],
    )
    return Path(root) / alias


def load_alias(
    directory: Path, version: str, variant: str, threads: int, alias: str
) -> GuardModel:  # pragma: no cover - needs onnxruntime + weights
    from breachlab.common.config import ServeConfig
    from breachlab.serve.onnx_backend import load_guard

    return load_guard(
        directory, ServeConfig(), version, threads=threads, file=VARIANT_FILES[variant]
    )


def effective_variant(directory: Path, requested: str) -> str:
    """The requested variant if its file is present, else fp32 (not every alias ships every variant)."""
    if (directory / VARIANT_FILES[requested]).exists():
        return requested
    if (directory / VARIANT_FILES["fp32"]).exists():
        return "fp32"
    return requested  # let the loader report the missing file


def version_of(directory: Path, alias: str) -> str:
    """Version string = the exported model (``export.json`` ``which``, e.g. ``sft-0.6b-s0``) or the alias."""
    try:
        meta = json.loads((directory / "export.json").read_text(encoding="utf-8"))
        return str(meta.get("which") or alias)
    except (OSError, ValueError):
        return alias


def boot(
    s: Settings,
    state: State,
    download: Download = download_alias,
    loader: Loader = load_alias,
) -> None:
    """Load every alias that exists; the champion is mandatory. Records timings in ``state.boot``."""
    state.variant = s.variant
    models: dict[str, GuardModel] = {}
    try:
        for alias in s.aliases:
            t0 = time.perf_counter()
            try:
                directory = s.model_dir / alias if s.model_dir else download(s, alias)
                if not directory.is_dir():
                    raise FileNotFoundError(f"{alias}/ not found")
                state.boot.download_s[alias] = time.perf_counter() - t0
                t1 = time.perf_counter()
                version = version_of(directory, alias)
                variant = effective_variant(directory, s.variant)
                models[alias] = loader(directory, version, variant, s.threads, alias)
                if s.max_tokens:
                    models[alias].max_tokens = {
                        "input_guard": s.max_tokens,
                        "output_filter": min(
                            s.max_tokens, models[alias].max_tokens["output_filter"]
                        ),
                    }
            except Exception as exc:  # noqa: BLE001 - optional aliases may be absent or broken
                if alias == "champion":
                    raise
                log.warning("alias %s unavailable: %s", alias, type(exc).__name__)
                continue
            state.boot.load_s[alias] = time.perf_counter() - t1
            state.versions[alias] = version
            state.variants[alias] = variant
            log.info("loaded %s (%s) in %.1fs", alias, version, state.boot.load_s[alias])
        state.service = GuardService(models)
    except Exception as exc:  # noqa: BLE001 - surfaced on /health and /info, never silently
        state.boot.error = f"{type(exc).__name__}: {exc}"
        log.error("boot failed: %s", state.boot.error)
    finally:
        state.boot.total_s = time.perf_counter() - state.boot.started


def boot_with_retry(
    s: Settings,
    state: State,
    download: Download = download_alias,
    loader: Loader = load_alias,
    attempts: int = 3,
    wait_s: tuple[float, ...] = (15.0, 45.0),
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """``boot`` again after a transient failure (Hub 5xx, rate limit); the last error stays on /health."""
    for i in range(attempts):
        state.boot.error = None
        boot(s, state, download, loader)
        if state.service is not None or i == attempts - 1:
            return
        sleep(wait_s[min(i, len(wait_s) - 1)])


def create_app(
    settings: Settings | None = None,
    download: Download = download_alias,
    loader: Loader = load_alias,
    background: bool = True,
) -> FastAPI:
    state = State()
    s = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        if background and not s.boot_sync:
            threading.Thread(
                target=boot_with_retry,
                args=(s, state, download, loader),
                daemon=True,
                name="boot",
            ).start()
        else:
            boot(s, state, download, loader)
        yield

    def authorized(authorization: str | None = Header(default=None)) -> None:
        """Bearer check on /score when GUARD_TOKEN is set (the gateway sends its own GUARD_TOKEN)."""
        if s.auth_token is None:
            return
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), s.auth_token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    app = FastAPI(title="BreachLab guard", version="1", lifespan=lifespan)
    app.state.guard = state

    @app.get("/health")
    def health() -> Any:
        from fastapi.responses import JSONResponse

        if state.service is None:
            return JSONResponse(
                {"ok": False, "models": {}, "error": state.boot.error}, status_code=503
            )
        return {"ok": True, "models": dict(state.versions)}

    @app.get("/info")
    def info() -> dict[str, Any]:
        ort = None
        try:  # pragma: no cover - environment dependent
            import onnxruntime

            ort = onnxruntime.__version__
        except ImportError:
            pass
        b = state.boot
        return {
            "variant": state.variant,
            "variants": dict(state.variants),
            "threads": s.threads,
            "onnxruntime": ort,
            "ready": state.service is not None,
            "download_s": b.download_s,
            "load_s": b.load_s,
            "cold_start_s": b.total_s,
            "error": b.error,
        }

    @app.post(
        "/score",
        response_model=ScoreResponse,
        response_model_exclude_none=True,
        dependencies=[Depends(authorized)],
    )
    def score(req: ScoreRequest) -> ScoreResponse:
        svc = state.service
        if svc is None:
            raise HTTPException(status_code=503, detail="guard is loading")
        try:
            with state.lock:
                r = svc.score(req.text, req.kind, req.alias)
        except Exception as exc:  # noqa: BLE001 - fail closed for the gateway (503), never a fake SAFE
            log.error("score failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="guard error") from exc
        v = r.verdict
        return ScoreResponse(
            label=v.label,
            category=v.category,
            p_unsafe=v.p_unsafe,
            confidence=v.p_unsafe,
            version=r.version,
            latency_ms=r.latency_ms,
            stage=r.stage,
        )

    return app
