"""Guard on a Vercel Python function (ADR-0080): the frozen /score + /health contract, weights fetched into RAM.

The Space service (``breachlab.serve.space_app``) is reused as is: same request validation, bearer check, scoring
maths and response model. Only two things differ on a serverless host:

* nothing persists, so the single-file int4 ONNX is streamed from the private Hub repo into memory at cold start
  (``HF_TOKEN``) and the session is built from those bytes; the small companions (tokenizer, calibration, prefix K/V)
  go to ``/tmp``;
* a background thread started at boot may be frozen as soon as the response is sent, so the models are loaded
  inside the first request that finds none (every route, ``/health`` included), under a lock, instead of in a
  lifespan thread. A request that arrives while the load runs waits for it.

Configuration (environment): ``HF_TOKEN`` (read access to the model repo), ``GUARD_TOKEN`` (optional bearer on
``/score``), ``GUARD_MODEL_REPO``, ``GUARD_ONNX_FILE`` (file under ``<GUARD_ALIAS>/``, default ``champion/``), ``GUARD_THREADS``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("BREACHLAB_CONFIG_DIR", str(HERE / "configs"))

import httpx  # noqa: E402
from fastapi import Request, Response  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from breachlab.serve import space_app as sa  # noqa: E402
from breachlab.serve.guard_runtime import GuardModel  # noqa: E402

IMPORTED_AT = time.time()
ALIAS = "champion"  # the alias this host serves; requests for other aliases fall back to it
# Hub folder for the graph and its companions: champion/ by default, base/ for the L3 host (ADR-0086, ADR-0090).
HUB_DIR = os.environ.get("GUARD_ALIAS", ALIAS)
ONNX_FILE = os.environ.get("GUARD_ONNX_FILE", "guard-q4b32.onnx")
# The int4 graph has its own threshold (recalibrated on val, ADR-0082); the fp32 calibration.json stays untouched.
CALIBRATION_FILE = os.environ.get("GUARD_CALIBRATION_FILE", "calibration-q4b32.json")
# Cascade mode (ADR-0105): a Hub folder holding the int8 student; the 0.6B teacher in HUB_DIR is then loaded lazily on the
# first escalated request. Unset = the single-model host exactly as before.
CASCADE_DIR = os.environ.get("GUARD_CASCADE_DIR", "")
CASCADE_FILES = ("student-int8.onnx", "student-tokenizer.json", "student-meta.json")
SMALL_FILES = (
    "tokenizer.json",
    "calibration.json",
    "export.json",
    "prefix_kv_input.npz",
    "prefix_kv_output.npz",
)
WORK = Path(os.environ.get("GUARD_WORK_DIR", "/tmp/guard"))  # noqa: S108 - the only writable path on the host
CHUNK = 8 * 1024 * 1024

_blob: bytes | None = None
_fetch: dict[str, float] = {}
_boot_lock = threading.Lock()
_boot_lock_teacher = threading.Lock()
_served = 0


def _url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _get(client: httpx.Client, repo: str, path: str, token: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.send(client.build_request("GET", _url(repo, path), headers=headers), stream=True)


def remote_name(name: str) -> str:
    """The Hub file behind a local companion name (only ``calibration.json`` is variant specific)."""
    return CALIBRATION_FILE if name == "calibration.json" else name


def fetch_small(client: httpx.Client, s: sa.Settings, name: str, dest: Path) -> None:
    r = _get(client, s.repo, f"{HUB_DIR}/{remote_name(name)}", s.token)
    try:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
    finally:
        r.close()


def fetch_blob(client: httpx.Client, s: sa.Settings) -> bytes:
    """Stream the ONNX file into RAM. Chunks are joined once at the end (peak = file size x 2 for an instant)."""
    r = _get(client, s.repo, f"{HUB_DIR}/{ONNX_FILE}", s.token)
    try:
        r.raise_for_status()
        chunks = [c for c in r.iter_bytes(CHUNK)]
    finally:
        r.close()
    data = b"".join(chunks)
    del chunks
    return data


def download_teacher(s: sa.Settings) -> Path:
    """``space_app.Download`` for a serverless host: bytes to RAM, small files to /tmp, a placeholder for the graph.

    ``space_app.effective_variant`` looks for ``guard-kv.onnx`` in the returned directory to pick the prefix-KV
    graph; the real graph is in memory, so an empty placeholder file of that name satisfies the check.
    """
    global _blob
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
        for name in SMALL_FILES:
            fetch_small(client, s, name, WORK / name)
        _fetch["small_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        _blob = fetch_blob(client, s)
        _fetch["onnx_s"] = time.perf_counter() - t1
    _fetch["onnx_mb"] = len(_blob) / 1e6
    _fetch["onnx_mb_per_s"] = _fetch["onnx_mb"] / max(_fetch["onnx_s"], 1e-9)
    (WORK / "guard-kv.onnx").write_bytes(b"")
    return WORK


def load_teacher_model(
    directory: Path, version: str, variant: str, threads: int, alias: str
) -> GuardModel:
    """``space_app.Loader``: one ORT session over the in-memory graph (prefix-KV path)."""
    global _blob
    from breachlab.common.config import ServeConfig
    from breachlab.serve.onnx_backend import KV_FILE, load_guard

    blob, _blob = (
        _blob,
        None,
    )  # drop our reference: ORT copies what it needs, the bytes can be freed
    if blob is None:
        raise RuntimeError("model bytes missing")
    try:
        return load_guard(
            directory, ServeConfig(), version, threads=threads, file=KV_FILE, model_bytes=blob
        )
    finally:
        del blob


def download_cascade(s: sa.Settings) -> Path:
    """Cascade mode: only the student (about 100 MB) and the teacher's ``export.json`` are fetched at boot."""
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "cascade").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
        fetch_small(client, s, "export.json", WORK / "export.json")
        for name in CASCADE_FILES:
            r = _get(client, s.repo, f"{CASCADE_DIR}/{name}", s.token)
            try:
                r.raise_for_status()
                with (WORK / "cascade" / name).open("wb") as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
            finally:
                r.close()
    _fetch["student_s"] = time.perf_counter() - t0
    (WORK / "guard-kv.onnx").write_bytes(b"")
    return WORK


def load_cascade(
    directory: Path, version: str, variant: str, threads: int, alias: str
) -> GuardModel:
    import json

    from breachlab.common.config import ServeConfig
    from breachlab.serve.cascade import CascadeModel, CascadeParams, StudentScorer

    meta = json.loads((directory / "cascade" / "student-meta.json").read_text())
    scorer = StudentScorer(
        str(directory / "cascade" / "student-int8.onnx"),
        str(directory / "cascade" / "student-tokenizer.json"),
        meta,
        threads,
    )
    params = CascadeParams(
        str(meta["version"]),
        float(meta["temperature"]),
        float(meta["t_low"]),
        float(meta["t_high"]),
    )

    def teacher() -> GuardModel:
        # first escalated request: fetch and load the 0.6B (about 10 s once per instance)
        with _boot_lock_teacher:
            t0 = time.perf_counter()
            d = download_teacher(S)
            m = load_teacher_model(d, version, "kv", threads, alias)
            _fetch["teacher_load_s"] = time.perf_counter() - t0
            return m

    return CascadeModel(scorer, teacher, params, ServeConfig().max_tokens.model_dump())


def download(s: sa.Settings, alias: str) -> Path:
    return download_cascade(s) if CASCADE_DIR else download_teacher(s)


def load(directory: Path, version: str, variant: str, threads: int, alias: str) -> GuardModel:
    if CASCADE_DIR:
        return load_cascade(directory, version, variant, threads, alias)
    return load_teacher_model(directory, version, variant, threads, alias)


def settings() -> sa.Settings:
    env = dict(os.environ)
    env.setdefault("GUARD_ALIASES", ALIAS)
    env.setdefault("GUARD_VARIANT", "kv")
    return sa.Settings.from_env(env)


S = settings()
app = sa.create_app(
    S,
    download=lambda s, alias: download(s, alias),  # looked up at call time (tests replace them)
    loader=lambda *a: load(*a),
    background=False,
)


@asynccontextmanager
async def _no_lifespan(_: Any) -> Any:
    yield  # models load in the first request instead (see module docstring)


app.router.lifespan_context = _no_lifespan
STATE: sa.State = app.state.guard


def ensure_loaded() -> None:
    with _boot_lock:
        if STATE.service is None:
            STATE.boot = sa.Boot()
            sa.boot(S, STATE, lambda s, a: download(s, a), lambda *a: load(*a))


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


@app.middleware("http")
async def lazy_boot(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    global _served
    cold = _served == 0
    t0 = time.perf_counter()
    if STATE.service is None:
        await run_in_threadpool(ensure_loaded)
    _served += 1
    response = await call_next(request)
    response.headers["x-guard-cold"] = "1" if cold else "0"
    response.headers["x-guard-boot-s"] = f"{STATE.boot.total_s or 0.0:.3f}"
    response.headers["x-guard-wall-ms"] = f"{(time.perf_counter() - t0) * 1000.0:.1f}"
    return response


def _cascade_info() -> dict[str, Any] | None:
    svc = STATE.service
    model = svc.models.get("champion") if svc is not None else None
    counts = getattr(model, "counts", None)
    if counts is None:
        return None
    return {"counts": dict(counts), "escalation_rate": model.escalation_rate()}  # type: ignore[union-attr]


@app.get("/vercel-info")
def vercel_info() -> dict[str, Any]:
    return {
        "region": os.environ.get("VERCEL_REGION"),
        "python": sys.version.split()[0],
        "cpu_limit": sa.cpu_limit(),
        "threads": S.threads,
        "rss_mb": rss_mb(),
        "instance_age_s": time.time() - IMPORTED_AT,
        "served": _served,
        "fetch": dict(_fetch),
        "onnx_file": ONNX_FILE,
        "boot_error": STATE.boot.error,
        "cascade": _cascade_info(),
    }
