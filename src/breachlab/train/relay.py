"""Transit-only relay: Kaggle kernel output -> Hugging Face Hub, without ever touching the bytes (ADR-0040).

The Kaggle API has no way to hand a kernel a secret (no ``secrets`` field in kernel-metadata, no REST endpoint;
user secrets are UI-only), so a kernel cannot push to the Hub itself. The sanctioned fallback is this relay in
the dev container: ``kaggle kernels output --file-pattern <regex>`` writes only the matching files into a temp
dir, each file is uploaded with ``huggingface_hub`` (multipart LFS from disk, never read into memory) and
deleted at once, and the temp dir is removed on exit. Nothing here imports torch, transformers or
onnxruntime; no model is loaded or run. Logs print file names and sizes only, never tokens.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPO = "Pseudo-Harsh/breachlab-guard-v1"
OWNER = "giriharsh"
ONNX_FILES = r"(guard-(fp32|w8|int8)\.onnx(\.data)?|tokenizer\.json|tokenizer_config\.json|calibration\.json|export\.json)"
KV_FILES = r"(guard-kv\.onnx(\.data)?|prefix_kv_(input|output)\.npz|kv_report\.json|calibration\.json|export\.json|tokenizer\.json)"
ADAPTER_FILES = r"(adapter_model\.safetensors|adapter_config\.json)"


class Uploader(Protocol):
    def upload_file(self, *, path_or_fileobj: str, path_in_repo: str, repo_id: str) -> Any: ...


Runner = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class RelayJob:
    """Fetch files of ``kernel`` whose path matches ``pattern`` and upload them below ``dest`` in the repo.

    ``strip`` is the path prefix (inside the kernel output) that is dropped before ``dest`` is prepended, so
    ``out/export-sft-0.6b-s0/guard-fp32.onnx`` with strip ``out/export-sft-0.6b-s0`` and dest ``champion``
    lands at ``champion/guard-fp32.onnx``.
    """

    kernel: str
    pattern: str
    strip: str
    dest: str


def repo_path(job: RelayJob, rel: str) -> str:
    """Map a path relative to the downloaded output root to its path in the repo."""
    rel = rel.replace("\\", "/")
    prefix = job.strip.strip("/") + "/"
    if job.strip and rel.startswith(prefix):
        rel = rel[len(prefix) :]
    elif job.strip:
        raise ValueError(f"{rel!r} is outside the expected prefix {job.strip!r}")
    return f"{job.dest.strip('/')}/{rel}" if job.dest else rel


def kaggle_runner(args: Sequence[str]) -> None:  # pragma: no cover - network
    out = subprocess.run(
        [sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, check=False
    )
    if out.returncode:
        # Tail only: the CLI never echoes credentials, but keep the message short anyway.
        raise RuntimeError((out.stdout + out.stderr)[-500:])


def fetch(job: RelayJob, dest: Path, run: Runner = kaggle_runner) -> list[Path]:
    """Download only the files matching the job's pattern into ``dest``; return what arrived."""
    re.compile(job.pattern)  # fail fast on a bad regex before spending a download
    run(["kernels", "output", job.kernel, "-p", str(dest), "--file-pattern", job.pattern])
    return sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix != ".log")


def relay_job(
    job: RelayJob,
    api: Uploader,
    repo: str = REPO,
    run: Runner = kaggle_runner,
    tmp_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Run one job end to end and return the repo paths uploaded. The temp dir never outlives the call."""
    if tmp_root is not None:
        tmp_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="relay-", dir=tmp_root))
    sent: list[str] = []
    try:
        files = fetch(job, tmp, run)
        if not files:
            raise FileNotFoundError(f"no files from {job.kernel} match {job.pattern!r}")
        for f in files:
            target = repo_path(job, f.relative_to(tmp).as_posix())
            api.upload_file(path_or_fileobj=str(f), path_in_repo=target, repo_id=repo)
            log(f"uploaded {target} ({f.stat().st_size} bytes)")
            f.unlink()  # free disk before the next (possibly multi-GB) file
            sent.append(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sent


def export_job(kernel: str, which: str, alias: str, files: str = ONNX_FILES) -> RelayJob:
    return RelayJob(kernel, rf".*export-{re.escape(which)}/{files}$", f"out/export-{which}", alias)


def adapter_job(kernel: str, name: str) -> RelayJob:
    return RelayJob(
        kernel,
        rf".*{re.escape(name)}/adapter/{ADAPTER_FILES}$",
        f"out/{name}/adapter",
        f"adapters/{name}",
    )


def guard_plan(kernels: dict[str, str] | None = None) -> dict[str, RelayJob]:
    """The layout the Space loads: champion/, challenger/, base/ plus adapters/ (see space/README.md)."""
    k = {
        "champion": f"{OWNER}/breachlab-export3-sft",
        "challenger": f"{OWNER}/breachlab-export3-dpo",
        "base": f"{OWNER}/breachlab-export3-base",
        **(kernels or {}),
    }
    return {
        "champion": export_job(k["champion"], "sft-0.6b-s0", "champion"),
        "challenger": export_job(k["challenger"], "dpo-0.6b-s0", "challenger"),
        "base": export_job(k["base"], "base", "base"),
        "champion-kv": export_job(
            f"{OWNER}/breachlab-export4-sftkv", "sft-0.6b-s0", "champion", KV_FILES
        ),
        "challenger-kv": export_job(
            f"{OWNER}/breachlab-export4-dpokv", "dpo-0.6b-s0", "challenger", KV_FILES
        ),
        "base-kv": export_job(f"{OWNER}/breachlab-export4-basekv", "base", "base", KV_FILES),
        "adapter-sft-0.6b": adapter_job(f"{OWNER}/breachlab-sft-06b-s0", "sft-0.6b-s0"),
        "adapter-dpo-0.6b": adapter_job(f"{OWNER}/breachlab-dpo-06b-s0", "dpo-0.6b-s0"),
        "adapter-sft-1.7b": adapter_job(f"{OWNER}/breachlab-sft-17b-s0", "sft-1.7b-s0"),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    import argparse
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keep the log readable
    from huggingface_hub import HfApi

    plan = guard_plan()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="+", choices=sorted(plan))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument(
        "--tmp", type=Path, default=None, help="scratch dir with room for the largest file"
    )
    a = ap.parse_args(argv)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(a.repo, private=True, exist_ok=True)
    for name in a.names:
        relay_job(plan[name], api, a.repo, tmp_root=a.tmp)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
