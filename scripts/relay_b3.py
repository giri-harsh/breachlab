"""Transit-only relay of a quantized single-file guard from the Kaggle kernel output to the Hub (ADR-0040, ADR-0080).

    python scripts/relay_b3.py b3-quant q4b32          # -> Pseudo-Harsh/breachlab-guard-v1 champion/guard-q4b32.onnx
    python scripts/relay_b3.py b3b-gold2 q4b32 base    # -> base/guard-q4b32.onnx (the L3 host, ADR-0086)

Only the one matching `.onnx` file is fetched (``kaggle kernels output --file-pattern``), uploaded from disk with
``huggingface_hub`` and deleted at once; the temp dir is removed on exit. Weights are never loaded or run here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from breachlab.train.relay import OWNER, REPO, RelayJob, relay_job  # noqa: E402


def kaggle_bin(args: Sequence[str]) -> None:
    """The ``kaggle`` console script (``python -m kaggle`` is not runnable in kaggle>=2)."""
    exe = str(Path(sys.executable).with_name("kaggle"))
    out = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
    if out.returncode:
        raise RuntimeError((out.stdout + out.stderr)[-500:])


def job(slug: str, variant: str, dest: str = "champion") -> RelayJob:
    return RelayJob(
        f"{OWNER}/{slug}", rf".*b3/{variant}/guard-{variant}\.onnx$", f"b3/{variant}", dest
    )


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        raise SystemExit(__doc__)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    sent = relay_job(job(argv[0], argv[1], *argv[2:]), api, REPO, run=kaggle_bin)
    print("uploaded:", ", ".join(sent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
