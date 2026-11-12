"""Reproducibility tags (PRD 12): every model traceable to git SHA + ``data-vN`` + config hash + seed.

``repro_tags`` builds the tag dict, ``log_repro_tags`` sets it on the active MLflow run. The git SHA comes from
``BREACHLAB_GIT_SHA`` / ``GITHUB_SHA``, then a ``GIT_SHA`` file staged next to ``src/`` (a Kaggle kernel has no
``.git``; ``kaggle_job.stage_code`` writes it), then ``git rev-parse HEAD``, else ``"unknown"`` (never raises).
The config hash is sha256 over the resolved ``TrainConfig`` as canonical JSON, without ``data_dir`` and
``output_dir`` (they only say where files sit on a given machine, so the same run hashes the same everywhere).
No network, no secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from breachlab.common.config import TrainConfig

# The repo root locally; the staged dataset root on Kaggle.
PKG_ROOT = Path(__file__).resolve().parents[3]
SHA_FILE = "GIT_SHA"
LOCATION_FIELDS = frozenset({"data_dir", "output_dir"})
DATA_FILES = ("train.jsonl", "val.jsonl", "mining_pool.jsonl")
UNKNOWN = "unknown"


def _git_rev_parse(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def git_sha(
    env: Mapping[str, str] | None = None,
    root: Path = PKG_ROOT,
    rev_parse: Callable[[Path], str | None] = _git_rev_parse,
) -> str:
    env = os.environ if env is None else env
    for var in ("BREACHLAB_GIT_SHA", "GITHUB_SHA"):
        if env.get(var, "").strip():
            return env[var].strip()
    f = root / SHA_FILE
    if f.is_file():
        sha = f.read_text(encoding="utf-8").strip()
        if sha:
            return sha
    return rev_parse(root) or UNKNOWN


def config_payload(cfg: TrainConfig) -> dict[str, Any]:
    d = cfg.model_dump(mode="json")
    return {k: v for k, v in d.items() if k not in LOCATION_FIELDS}


def config_sha256(cfg: TrainConfig) -> str:
    blob = json.dumps(config_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def data_sha256(data_dir: Path) -> str | None:
    """sha256 over the split files that exist (name + bytes, fixed order); None when none exist."""
    h = hashlib.sha256()
    seen = False
    for name in DATA_FILES:
        p = Path(data_dir) / name
        if p.is_file():
            seen = True
            h.update(name.encode())
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest() if seen else None


def repro_tags(cfg: TrainConfig, sha: str | None = None) -> dict[str, str]:
    tags = {
        "git_sha": sha or git_sha(),
        "config_sha256": config_sha256(cfg),
        "data_version": f"data-{cfg.data_version}",
        "seed": str(cfg.seed),
        "base_model": cfg.base_model,
    }
    dsha = data_sha256(cfg.data_dir)
    if dsha:
        tags["data_sha256"] = dsha
    return tags


class TagSink(Protocol):
    def set_tags(self, tags: dict[str, Any]) -> None: ...


def log_repro_tags(
    cfg: TrainConfig, sink: TagSink | None = None, sha: str | None = None
) -> dict[str, str]:
    """Set the tags on the active MLflow run (``sink`` defaults to the ``mlflow`` module) and return them."""
    tags = repro_tags(cfg, sha)
    if sink is None:  # pragma: no cover - needs mlflow
        import mlflow

        mlflow.set_tags(tags)
    else:
        sink.set_tags(tags)
    return tags


def missing_tags(tags: Mapping[str, Any]) -> list[str]:
    """Required PRD 12 tags that are absent or unknown on a run (empty list = traceable)."""
    need = ("git_sha", "config_sha256", "data_version", "seed")
    return [k for k in need if str(tags.get(k, "")).strip() in ("", UNKNOWN)]
