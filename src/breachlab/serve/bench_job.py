"""Push a Kaggle CPU kernel that runs the Space app pinned to 2 cores and measures the latency ladder (ADR-0039).

The kernel mounts the export kernels' outputs (weights never leave Kaggle for this measurement) and executes
``breachlab.serve.bench_runner``. See that module for why this stands in for the final host.
"""

from __future__ import annotations

import json
from pathlib import Path

from breachlab.train.export_job import EXPORT_CODE_SLUG
from breachlab.train.kaggle_job import DATA_SLUG, OWNER, ROOT, kaggle, kernel_meta

BENCH_PIP = [
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic-settings",
    "pyyaml",
    "numpy",
    "scipy",
    "onnxruntime==1.31.1",
    "tokenizers",
]
DEFAULT_SOURCES = [
    "giriharsh/breachlab-export3-sft",
    "giriharsh/breachlab-export3-dpo",
    "giriharsh/breachlab-export3-base",
]


def bench_script(args: list[str]) -> str:
    return (
        "import glob, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{BENCH_PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "os.environ['PYTHONPATH'] = os.path.join(root, 'src')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "gold = glob.glob('/kaggle/input/**/gold_test.jsonl', recursive=True)[0]\n"
        "from breachlab.serve.bench_runner import main\n"
        f"sys.exit(main(['--gold', gold, *{args!r}]))\n"
    )


def push_bench(  # pragma: no cover - network
    slug: str, sources: list[str], args: list[str], workdir: Path
) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER,
        slug,
        [f"{OWNER}/{EXPORT_CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"],
        gpu=False,
        kernels=sources,
    )
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (workdir / "run.py").write_text(bench_script(args), encoding="utf-8")
    return kaggle("kernels", "push", "-p", str(workdir))


if __name__ == "__main__":  # pragma: no cover - network
    import argparse

    from breachlab.train.export_job import stage_export_code
    from breachlab.train.kaggle_job import upload_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="must start with breachlab-export (CPU kernel)")
    ap.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    ap.add_argument("--variants", default="fp32")
    ap.add_argument("--parity-rows", type=int, default=0)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-code-upload", action="store_true")
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    if not a.skip_code_upload:
        stage_export_code(work / "export-code")
        upload_dataset(work / "export-code", EXPORT_CODE_SLUG)
        import time

        # A kernel pushed while the new dataset version is still processing mounts the previous one.
        while "ready" not in kaggle("datasets", "status", f"{OWNER}/{EXPORT_CODE_SLUG}"):
            time.sleep(10)
        time.sleep(60)
    kernel_args = [
        "--variants",
        a.variants,
        "--parity-rows",
        str(a.parity_rows),
        "--requests",
        str(a.requests),
        "--threads",
        str(a.threads),
    ]
    print(push_bench(a.slug, a.sources, kernel_args, work / a.slug))
