"""Assemble the Vercel project for the guard function (ADR-0080) from the repo.

``stage`` writes a self-contained folder: the function (``vercel-guard/``) plus the runtime modules vendored
byte-for-byte from ``src/breachlab`` (same list as the Space image, so the prompt and scoring maths cannot drift).
The folder carries no version-control metadata on purpose: a CLI deploy from a checkout whose HEAD author is not a
team member ends BLOCKED (ADR-0055). Deployment itself (project creation, env, ``vercel deploy``) is driven by
``scripts/deploy_vercel_guard.sh``; no secret is ever written to the staged folder.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from breachlab.serve.space_build import CONFIGS, ROOT, VENDOR

SRC = ROOT / "vercel-guard"
OWN_FILES = ["vercel.json", "requirements.txt", ".python-version", "api/index.py"]
BUNDLE_LIMIT_MB = (
    250.0  # Vercel's unzipped function size limit; the staged folder must stay far below
)


def stage(dest: Path, root: Path = ROOT, src: Path = SRC) -> list[str]:
    """Write the project into ``dest`` (emptied first) and return the relative paths written."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    written: list[str] = []
    for rel in OWN_FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        written.append(rel)
    for rel in VENDOR:
        # The function imports breachlab from the project root (api/index.py puts it on sys.path).
        target = dest / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "src" / rel, target)
        written.append(rel)
    (dest / "configs").mkdir()
    for name in CONFIGS:
        shutil.copy2(root / "configs" / name, dest / "configs" / name)
        written.append(f"configs/{name}")
    return written


def size_mb(folder: Path) -> float:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dest", type=Path)
    a = ap.parse_args(argv)
    files = stage(a.dest)
    print(
        f"staged {len(files)} files, {size_mb(a.dest):.2f} MB of code (limit {BUNDLE_LIMIT_MB} MB incl. deps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
