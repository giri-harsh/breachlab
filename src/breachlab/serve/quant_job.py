"""Push the Kaggle CPU kernel that builds and checks the single-file int4 / int8 guard (ADR-0080, lane B3).

The kernel mounts the champion prefix-KV export kernel (weights never leave Kaggle for the quantization and the
parity run), the lane's own code dataset (``b3-code``; the shared datasets of other lanes are never touched) and the
data dataset. Fetch results with ``kaggle kernels output <slug> --file-pattern '.*\\.(log|json|npy)$'``; the chosen
ONNX file is relayed to the Hub by ``relay_b3`` (transit only, ADR-0040), adapters are never downloaded.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from breachlab.serve.bench_job import BENCH_PIP
from breachlab.train.kaggle_job import DATA_SLUG, OWNER, ROOT, dataset_meta, kernel_meta

CODE_SLUG = "b3-code"
SOURCE_KERNEL = (
    "giriharsh/breachlab-export4-sftkv"  # holds out/export-sft-0.6b-s0/guard-kv.onnx (+ prefix npz)
)
PIP = [*BENCH_PIP, "onnx", "onnx-ir", "ml_dtypes"]


def kaggle(*args: str) -> str:  # pragma: no cover - network
    """The ``kaggle`` console script next to the interpreter (``python -m kaggle`` is not runnable in kaggle>=2)."""
    exe = str(Path(sys.executable).with_name("kaggle"))
    out = subprocess.run([exe, *args], capture_output=True, text=True, check=False)
    if out.returncode:
        raise RuntimeError((out.stdout + out.stderr)[-800:])
    return out.stdout


def quant_script(args: list[str], gold_name: str = "gold_test.jsonl") -> str:
    return (
        "import glob, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "os.environ['PYTHONPATH'] = os.path.join(root, 'src')  # the pinned probe subprocesses import breachlab\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        f"gold = glob.glob('/kaggle/input/**/{gold_name}', recursive=True)[0]\n"
        "from breachlab.serve.quant_export import main\n"
        f"sys.exit(main(['--gold', gold, *{args!r}]))\n"
    )


def stage_code(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", dest / "configs")
    (dest / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta(OWNER, CODE_SLUG, "breachlab b3 code")), encoding="utf-8"
    )


def push_quant(  # pragma: no cover - network
    slug: str,
    args: list[str],
    workdir: Path,
    source: str = SOURCE_KERNEL,
    gold_name: str = "gold_test.jsonl",
) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER, slug, [f"{OWNER}/{CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"], gpu=False, kernels=[source]
    )
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (workdir / "run.py").write_text(quant_script(args, gold_name), encoding="utf-8")
    return kaggle("kernels", "push", "-p", str(workdir))


if __name__ == "__main__":  # pragma: no cover - network
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="b3-quant")
    ap.add_argument("--variants", default="")
    ap.add_argument("--rows", type=int, default=-1)
    ap.add_argument("--skip-code-upload", action="store_true")
    ap.add_argument("--gold-name", default="gold_test.jsonl", help="split file in the data dataset")
    ap.add_argument(
        "--source",
        default=SOURCE_KERNEL,
        help="kernel holding the prefix-KV export to quantize (default: the champion)",
    )
    ap.add_argument(
        "--dir-name", default="", help="export folder in the source kernel (base: export-base)"
    )
    ap.add_argument(
        "--no-reference", action="store_true", help="variant logits only (recalibration)"
    )
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    if not a.skip_code_upload:
        stage_code(work / "b3-code")
        try:
            kaggle(
                "datasets",
                "version",
                "-p",
                str(work / "b3-code"),
                "-m",
                "update",
                "--dir-mode",
                "zip",
            )
        except RuntimeError:
            kaggle("datasets", "create", "-p", str(work / "b3-code"), "--dir-mode", "zip")
        # A kernel pushed while the new dataset version is still processing mounts the previous one.
        while "ready" not in kaggle("datasets", "status", f"{OWNER}/{CODE_SLUG}"):
            time.sleep(10)
        time.sleep(60)
    extra = ["--rows", str(a.rows)] + (["--variants", a.variants] if a.variants else [])
    extra += ["--no-reference"] if a.no_reference else []
    extra += ["--dir-name", a.dir_name] if a.dir_name else []
    print(push_quant(a.slug, extra, work / a.slug, source=a.source, gold_name=a.gold_name))
