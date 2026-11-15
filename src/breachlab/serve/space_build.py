"""Assemble the guard service image context and, optionally, deploy it as a Hugging Face Docker Space.

The image context is self-contained, so the runtime modules are *vendored* byte-for-byte from ``src/breachlab``
(same prompt, same scoring maths as training: no second implementation to drift). ``stage`` builds the folder
(any Docker host builds from it: ``--stage-only DIR``, see docs/deploy-guard.md); ``deploy`` pushes it to a Space
with ``huggingface_hub`` and waits for RUNNING (Docker Spaces need an HF PRO account, ADR-0063). Secrets are
never written to files: the model-repo token is set as a Space secret through the API, only if the caller asks.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SPACE_SRC = ROOT / "space"
SPACE_ID = "Pseudo-Harsh/breachlab-guard"

# Modules the Space imports at runtime, relative to src/. Anything else in the package stays out of the image.
VENDOR = [
    "breachlab/__init__.py",
    "breachlab/common/__init__.py",
    "breachlab/common/config.py",
    "breachlab/data/__init__.py",
    "breachlab/data/schema.py",
    "breachlab/train/__init__.py",
    "breachlab/train/prompts.py",
    "breachlab/train/scoring.py",
    "breachlab/serve/__init__.py",
    "breachlab/serve/guard_runtime.py",
    "breachlab/serve/onnx_backend.py",
    "breachlab/serve/space_app.py",
    "breachlab/serve/cascade.py",
]
OWN_FILES = ["Dockerfile", "README.md", "requirements.txt", "app.py"]
CONFIGS = ["serve.yaml"]


def stage(dest: Path, root: Path = ROOT, space_src: Path = SPACE_SRC) -> list[str]:
    """Write the Space repo layout into ``dest`` (emptied first); return the relative paths written."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    written: list[str] = []
    for rel in OWN_FILES:
        shutil.copy2(space_src / rel, dest / rel)
        written.append(rel)
    for rel in VENDOR:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "src" / rel, target)
        written.append(rel)
    (dest / "configs").mkdir()
    for name in CONFIGS:
        shutil.copy2(root / "configs" / name, dest / "configs" / name)
        written.append(f"configs/{name}")
    return written


def wait_running(
    api: Any,
    repo: str,
    timeout_s: float = 1800,
    poll_s: float = 15,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> str:  # pragma: no cover - network
    """Poll the Space runtime until RUNNING (returns the stage) or a terminal failure / timeout (raises)."""
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        stage_name = str(api.get_space_runtime(repo).stage)
        if stage_name != last:
            log(f"space stage: {stage_name}")
            last = stage_name
        if stage_name == "RUNNING":
            return stage_name
        if stage_name in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            raise RuntimeError(f"space failed: {stage_name}")
        sleep(poll_s)
    raise TimeoutError(f"space not RUNNING after {timeout_s}s (last {last})")


def deploy(  # pragma: no cover - network
    space_id: str = SPACE_ID,
    private: bool = False,
    variables: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
    work: Path | None = None,
) -> str:
    """Create/update the Space, set variables and secrets, wait RUNNING.

    ``secrets`` maps a Space secret name to the *local environment variable* holding its value, so the deploy
    token (write access) and the token the Space runs with (read-only, see docs/deploy-guard.md) stay separate.
    """
    import os
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    for k, v in (variables or {}).items():
        api.add_space_variable(space_id, k, v)
    for name, env_name in (secrets or {}).items():
        api.add_space_secret(space_id, name, os.environ[env_name])  # value never printed
    folder = work or Path(tempfile.mkdtemp(prefix="space-"))
    stage(folder)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=space_id,
        repo_type="space",
        commit_message="deploy guard service",
    )
    wait_running(api, space_id)
    return f"https://{space_id.replace('/', '-').lower()}.hf.space"


def parse_secrets(specs: Sequence[str]) -> dict[str, str]:
    """``["HF_TOKEN=HF_READ_TOKEN", "GUARD_TOKEN"]`` -> ``{"HF_TOKEN": "HF_READ_TOKEN", "GUARD_TOKEN": "GUARD_TOKEN"}``."""
    out: dict[str, str] = {}
    for spec in specs:
        name, _, env_name = spec.partition("=")
        out[name] = env_name or name
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage-only",
        type=Path,
        metavar="DIR",
        help="write the self-contained Docker build context to DIR and exit (no network, no token)",
    )
    ap.add_argument("--space", default=SPACE_ID)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--model-repo", default="Pseudo-Harsh/breachlab-guard-v1")
    ap.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="NAME[=ENVVAR]",
        help="set Space secret NAME from local env var ENVVAR (default: NAME); repeatable. "
        "Example: --secret HF_TOKEN=HF_READ_TOKEN --secret GUARD_TOKEN",
    )
    ap.add_argument(
        "--var", action="append", default=[], metavar="KEY=VALUE", help="extra Space variable"
    )
    a = ap.parse_args(argv)
    if a.stage_only:
        print(f"staged {len(stage(a.stage_only))} files in {a.stage_only}")
        return 0
    print(  # pragma: no cover - network
        deploy(
            a.space,
            a.private,
            {
                "GUARD_VARIANT": a.variant,
                "GUARD_MODEL_REPO": a.model_repo,
                **dict(v.split("=", 1) for v in a.var),
            },
            parse_secrets(a.secret),
        )
    )
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
