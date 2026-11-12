"""Run the real Space app on a CPU box pinned to N cores and measure the latency ladder end to end.

Used in a Kaggle CPU kernel as a stand-in for a 2-vCPU container host (no free Docker host exists, ADR-0063; the
final host must be re-measured): the server is ``breachlab.serve.space_app`` started
with uvicorn on cores {0,1}, the client runs on the remaining cores, and the same ``space_bench.bench`` that
drives a deployed service drives it over localhost. Weights come from mounted kernel outputs, so no token is needed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from breachlab.data.schema import Row
from breachlab.eval.metrics import Pred, summarize
from breachlab.serve.space_bench import bench, clip, http_post, read_gold

# alias -> name of the export directory inside a kernel output
ALIAS_DIRS = {
    "champion": "export-sft-0.6b-s0",
    "challenger": "export-dpo-0.6b-s0",
    "base": "export-base",
}


def merge_dirs(sources: Sequence[Path], dest: Path) -> list[str]:
    """Symlink every file of ``sources`` into ``dest`` (first source wins on a name clash)."""
    dest.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for src in sources:
        for f in sorted(src.iterdir()):
            target = dest / f.name
            if f.is_file() and not target.exists():
                target.symlink_to(f.resolve())
                linked.append(f.name)
    return linked


def collect_models(input_root: Path, dest: Path) -> dict[str, list[str]]:
    """Build dest/<alias>/ from every ``export-*`` directory found under the mounted kernel outputs."""
    out: dict[str, list[str]] = {}
    for alias, dirname in ALIAS_DIRS.items():
        found = sorted(p for p in input_root.rglob(dirname) if p.is_dir())
        if found:
            out[alias] = merge_dirs(found, dest / alias)
    return out


def pinned(cpus: Sequence[int]) -> Any:
    def apply() -> None:  # pragma: no cover - runs in the forked child
        os.sched_setaffinity(0, set(cpus))

    return apply


def start_server(  # pragma: no cover - subprocess
    models: Path,
    variant: str,
    threads: int,
    port: int,
    cpus: Sequence[int],
    log: Path,
    max_tokens: int | None = None,
) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "GUARD_MODEL_DIR": str(models),
        "GUARD_VARIANT": variant,
        "GUARD_THREADS": str(threads),
        **({"GUARD_MAX_TOKENS": str(max_tokens)} if max_tokens else {}),
    }
    fh = log.open("ab")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "breachlab.serve.space_app:create_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        preexec_fn=pinned(cpus),
    )


def wait_ready(url: str, timeout_s: float = 600) -> float:  # pragma: no cover - network
    """Seconds until /health answers 200 (the cold start measured from process start)."""
    import httpx

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            if httpx.get(f"{url}/health", timeout=2.0).status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise TimeoutError("server did not become ready")


def quality(rows: list[Row], outputs: list[dict[str, Any]]) -> dict[str, float]:
    """AUROC / FPR / ASR / F1 from the service's own verdicts (p_unsafe, label), so the full serving path counts."""
    preds = [
        Pred(
            y_true=int(r.label == "UNSAFE"),
            y_pred=int(o["label"] == "UNSAFE"),
            p_unsafe=float(o["p_unsafe"]),
            cat_true=r.category,
            cat_pred=str(o["category"]),
            hard_benign=r.attack_family == "hard_benign",
        )
        for r, o in zip(rows, outputs, strict=True)
    ]
    return summarize(preds)


def split_spec(spec: str) -> tuple[str, int | None]:
    """``kv@256`` -> (``kv``, 256): a graph variant with a message token limit (ADR-0039 rung 4)."""
    name, _, tok = spec.partition("@")
    return name, int(tok) if tok else None


def run_variant(  # pragma: no cover - subprocess + network
    models: Path,
    spec: str,
    rows: list[Row],
    *,
    threads: int,
    server_cpus: Sequence[int],
    requests: int,
    concurrencies: Sequence[int],
    parity_rows: int,
    port: int,
    workdir: Path,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}"
    t0 = time.perf_counter()
    variant, max_tokens = split_spec(spec)
    proc = start_server(
        models,
        variant,
        threads,
        port,
        server_cpus,
        workdir / f"server-{spec}.log",
        max_tokens,
    )
    try:
        ready = wait_ready(url)
        res: dict[str, Any] = {
            "variant": variant,
            "max_tokens": max_tokens,
            "threads": threads,
            "cold_start_s": ready,
        }
        import httpx

        res["info"] = httpx.get(f"{url}/info", timeout=5).json()
        post = http_post(url)
        texts = clip([r.text for r in rows])
        for c in concurrencies:
            res[f"c{c}"] = asdict(bench(post, texts, requests=requests, concurrency=c))
        if parity_rows:
            sample = rows if parity_rows < 0 else rows[:parity_rows]
            outs = [post(t) for t in clip([r.text for r in sample])]
            res["quality"] = quality(sample, outs)
            res["quality_n"] = len(sample)
        res["wall_s"] = time.perf_counter() - t0
        return res
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - kernel entry
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working/bench"))
    ap.add_argument("--models-dir", type=Path, default=Path("/tmp/guard-models"))
    ap.add_argument("--variants", default="fp32")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--server-cpus", default="0,1")
    ap.add_argument("--client-cpus", default="2,3")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--parity-rows", type=int, default=0)
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args(argv)
    server_cpus = [int(x) for x in a.server_cpus.split(",")]
    os.sched_setaffinity(0, {int(x) for x in a.client_cpus.split(",")})
    a.work.mkdir(parents=True, exist_ok=True)
    models = a.models_dir  # outside /kaggle/working: the symlinks must not become kernel output
    print("models:", json.dumps(collect_models(a.input_root, models)))
    rows = read_gold(a.gold)
    results: dict[str, Any] = {"cpus_total": os.cpu_count(), "server_cpus": server_cpus}
    for variant in a.variants.split(","):
        results[variant] = run_variant(
            models,
            variant,
            rows,
            threads=a.threads,
            server_cpus=server_cpus,
            requests=a.requests,
            concurrencies=a.concurrency,
            parity_rows=a.parity_rows,
            port=a.port,
            workdir=a.work,
        )
        (a.work / "bench.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps({variant: results[variant]}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
