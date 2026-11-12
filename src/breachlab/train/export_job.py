"""Push a Kaggle *CPU* kernel that exports one guard (champion / challenger / base) to slim ONNX (ADR-0040).

Separate from ``kaggle_job`` so it can pass export options (``which``, variants, parity rows) and use its own
code dataset (``breachlab-export-code``): re-uploading the shared ``breachlab-code`` dataset would swap the code
under another lane's kernels. Nothing secret is packaged. Results come back with
``kaggle kernels output --file-pattern`` and are relayed to the HF model repo by ``breachlab.train.relay``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from breachlab.train.kaggle_job import (
    DATA_SLUG,
    OWNER,
    PIP,
    ROOT,
    dataset_meta,
    kaggle,
    kernel_meta,
    upload_dataset,
)

EXPORT_CODE_SLUG = "breachlab-export-code"

# which -> Kaggle kernel whose output holds its adapter (None: untuned base, no adapter needed).
ADAPTER_KERNELS: dict[str, str | None] = {
    "sft-0.6b-s0": "giriharsh/breachlab-sft-06b-s0",
    "dpo-0.6b-s0": "giriharsh/breachlab-dpo-06b-s0",
    "base": None,
}


def export_script(which: str, model: str, variants: tuple[str, ...], parity_rows: int) -> str:
    """Kernel entry script: install deps, find the mounted code/data, call ``pipeline.run_export``."""
    return (
        "import glob, json, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "data = os.path.dirname(glob.glob('/kaggle/input/**/gold_test.jsonl', recursive=True)[0])\n"
        "from pathlib import Path\n"
        "from breachlab.common.config import ServeConfig, TrainConfig\n"
        "from breachlab.train.pipeline import run_export\n"
        f"cfg = TrainConfig(base_model={model!r}, seed=0, data_dir=Path(data), "
        "output_dir=Path('/kaggle/working/out'))\n"
        f"res = run_export(cfg, ServeConfig(), Path('/kaggle/input'), which={which!r}, "
        f"variants={variants!r}, parity_rows={parity_rows})\n"
        "print(json.dumps(res, default=str))\n"
    )


def kernels_for(which: str, override: str = "") -> list[str]:
    """Source kernels to mount: the one holding the adapter and its results.json (none for the base).

    ``override`` replaces the registered kernel, for a retrained adapter that keeps its name (guard v2).
    """
    src = override or ADAPTER_KERNELS[which]
    return [src] if src else []


def stage_export_code(dest: Path) -> None:
    """Copy src/breachlab and configs into the dataset folder of the export code."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", dest / "configs")
    (dest / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta(OWNER, EXPORT_CODE_SLUG, "breachlab export code")), encoding="utf-8"
    )


def push_export(  # pragma: no cover - network
    slug: str,
    which: str,
    model: str,
    variants: tuple[str, ...],
    parity_rows: int,
    workdir: Path,
    from_kernel: str = "",
) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER,
        slug,
        [f"{OWNER}/{EXPORT_CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"],
        gpu=False,
        kernels=kernels_for(which, from_kernel),
    )
    (workdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (workdir / "run.py").write_text(
        export_script(which, model, variants, parity_rows), encoding="utf-8"
    )
    return kaggle("kernels", "push", "-p", str(workdir))


if __name__ == "__main__":  # pragma: no cover - network
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="must start with breachlab-export (CPU kernel)")
    ap.add_argument("--which", required=True, choices=sorted(ADAPTER_KERNELS))
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--variants", default="fp32,w8")
    ap.add_argument("--parity-rows", type=int, default=0, help="gold rows to score (-1 all)")
    ap.add_argument("--skip-code-upload", action="store_true")
    ap.add_argument(
        "--from-kernel", default="", help="kernel holding the adapter (default: the registered one)"
    )
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
    print(
        push_export(
            a.slug,
            a.which,
            a.model,
            tuple(a.variants.split(",")),
            a.parity_rows,
            work / a.slug,
            a.from_kernel,
        )
    )
