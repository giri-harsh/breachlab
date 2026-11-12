"""Push the multi-model Kaggle CPU export kernel (SFT 0.6B + 1.7B: parity logits + latency on identical hardware).

    uv run python scripts/run_kaggle_export.py --slug breachlab-export3-sft \
        --spec sft-0.6b-s0=Qwen/Qwen3-0.6B@giriharsh/breachlab-sft-06b-s0 \
        --spec sft-1.7b-s0=Qwen/Qwen3-1.7B@giriharsh/breachlab-sft-17b-s0

The slug must start with `breachlab-export` so `kaggle_job.push_kernel` conventions pick a CPU kernel. The kernel
body is `scripts/kernel_export_body.py` (standalone; the lane that owns src/breachlab/train is not touched).
Fetch results with `kaggle kernels output <ref> -p DIR --file-pattern '.*\\.(log|json|npy)$'` (never weights).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from breachlab.train import kaggle_job

HERE = Path(__file__).resolve().parent
BODY = HERE / "kernel_export_body.py"


def parse_spec(spec: str) -> tuple[str, str, str]:
    """`which=base_model@owner/kernel` -> (which, base_model, kernel)."""
    which, rest = spec.split("=", 1)
    base, kernel = rest.rsplit("@", 1)
    return which, base, kernel


def prelude() -> str:
    """Install deps, find the code + data datasets, expose DATA_DIR (mirrors kaggle_job.kernel_script)."""
    return (
        "import glob, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{kaggle_job.PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "os.environ['DATA_DIR'] = os.path.dirname(glob.glob('/kaggle/input/**/gold_test.jsonl', recursive=True)[0])\n"
    )


def run_script(
    specs: list[tuple[str, str]], requests: int, keep_onnx: list[str], body: str | None = None
) -> str:
    text = body if body is not None else BODY.read_text("utf-8")
    header = f"SPECS = {specs!r}\nREQUESTS = {requests}\nKEEP_ONNX = {keep_onnx!r}\n"
    return prelude() + header + text


def build(
    slug: str, specs: list[tuple[str, str, str]], requests: int, keep_onnx: list[str], workdir: Path
) -> Path:
    if not slug.startswith("breachlab-export"):
        raise ValueError("slug must start with breachlab-export (CPU kernel)")
    workdir.mkdir(parents=True, exist_ok=True)
    meta = kaggle_job.kernel_meta(
        kaggle_job.OWNER,
        slug,
        [
            f"{kaggle_job.OWNER}/{kaggle_job.CODE_SLUG}",
            f"{kaggle_job.OWNER}/{kaggle_job.DATA_SLUG}",
        ],
        gpu=False,
        kernels=[k for _, _, k in specs],
    )
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (workdir / "run.py").write_text(
        run_script([(w, b) for w, b, _ in specs], requests, keep_onnx), encoding="utf-8"
    )
    return workdir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--keep-onnx", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    specs = [parse_spec(s) for s in a.spec]
    wd = build(a.slug, specs, a.requests, a.keep_onnx, Path(tempfile.mkdtemp(prefix=a.slug)))
    if a.dry_run:
        print(wd)
        return 0
    print(kaggle_job.kaggle("kernels", "push", "-p", str(wd)))  # pragma: no cover - network
    return 0


if __name__ == "__main__":
    sys.exit(main())
