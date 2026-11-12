"""Push the Kaggle CPU kernel that measures the quality of the message-truncation rungs (``trunc_eval``).

The kernel mounts the champion prefix-KV export kernel output (weights never leave Kaggle), the export code
dataset (``breachlab-export-code``, Lane B's own; the shared ``breachlab-code`` is never touched) and the data
dataset, and writes ``truncation_quality.json`` plus small logits ``.npy`` files. Fetch them with
``kaggle kernels output <slug> --file-pattern '.*\\.(log|json|npy)$'`` (never the ONNX files).
"""

from __future__ import annotations

import json
from pathlib import Path

from breachlab.serve.bench_job import BENCH_PIP
from breachlab.train.export_job import EXPORT_CODE_SLUG
from breachlab.train.kaggle_job import DATA_SLUG, OWNER, ROOT, kaggle, kernel_meta

SOURCE_KERNEL = (
    "giriharsh/breachlab-export4-sftkv"  # holds out/export-sft-0.6b-s0/guard-kv.onnx (+ prefix npz)
)
DEFAULT_SLUG = "breachlab-exportB2-trunc"


def trunc_script(args: list[str]) -> str:
    return (
        "import glob, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{BENCH_PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "gold = glob.glob('/kaggle/input/**/gold_test.jsonl', recursive=True)[0]\n"
        "from breachlab.serve.trunc_eval import main\n"
        f"sys.exit(main(['--gold', gold, *{args!r}]))\n"
    )


def push_trunc(  # pragma: no cover - network
    slug: str, args: list[str], workdir: Path, source: str = SOURCE_KERNEL
) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER,
        slug,
        [f"{OWNER}/{EXPORT_CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"],
        gpu=False,
        kernels=[source],
    )
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (workdir / "run.py").write_text(trunc_script(args), encoding="utf-8")
    return kaggle("kernels", "push", "-p", str(workdir))


if __name__ == "__main__":  # pragma: no cover - network
    import argparse
    import time

    from breachlab.train.export_job import stage_export_code
    from breachlab.train.kaggle_job import upload_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    ap.add_argument("--source", default=SOURCE_KERNEL)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--skip-code-upload", action="store_true")
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    if not a.skip_code_upload:
        stage_export_code(work / "export-code")
        upload_dataset(work / "export-code", EXPORT_CODE_SLUG)
        # A kernel pushed while the new dataset version is still processing mounts the previous one.
        while "ready" not in kaggle("datasets", "status", f"{OWNER}/{EXPORT_CODE_SLUG}"):
            time.sleep(10)
        time.sleep(60)
    print(push_trunc(a.slug, ["--n-boot", str(a.n_boot)], work / a.slug, a.source))
