"""Package code + data as private Kaggle datasets and push a GPU kernel (Modal GPU fallback, ADR-0020).

Nothing secret is packaged: the kernel only needs public HF model files; results come back through
``kaggle kernels output`` and are relayed to MLflow/HF by a Modal CPU function (``train/relay.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from breachlab.train.repro import git_sha

ROOT = Path(__file__).resolve().parents[3]
OWNER = "giriharsh"
CODE_SLUG = "breachlab-code"
DATA_SLUG = os.environ.get(
    "BREACHLAB_DATA_SLUG", "breachlab-data-v1"
)  # data-v2 runs set breachlab-data-v2
PIP = [
    "transformers>=4.51",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "pydantic-settings",
    "pyyaml",
    "pandera",
    "scipy",
    "torchao>=0.16.0",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "tokenizers",
]


def dataset_meta(owner: str, slug: str, title: str) -> dict[str, object]:
    return {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }


def kernel_meta(
    owner: str, slug: str, datasets: list[str], gpu: bool = True, kernels: list[str] | None = None
) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": True,
        "dataset_sources": datasets,
        "competition_sources": [],
        "kernel_sources": kernels or [],
    }


def kernel_script(model: str, seed: int, data_dir: str = "", extra: str = "") -> str:
    """Kernel entry script. Dataset mount paths vary (``/kaggle/input/<slug>`` or nested), so find them."""
    return (
        "import glob, os, shutil, subprocess, sys\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # one GPU: DPOTrainer rejects DataParallel\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "data = os.path.dirname(glob.glob('/kaggle/input/**/train.jsonl', recursive=True)[0])\n"
        "for prev in glob.glob('/kaggle/input/**/out', recursive=True):  # resume from a source kernel\n"
        "    shutil.copytree(prev, '/kaggle/working/out', dirs_exist_ok=True)\n"
        "from breachlab.train.kernel_main import main\n"
        f"sys.exit(main(['--model', {model!r}, '--seed', '{seed}', '--data-dir', data, "
        f"'--out', '/kaggle/working/out'{extra}]))\n"
    )


def stage_code(dest: Path) -> None:
    """Copy src/breachlab and configs into a dataset folder."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", dest / "configs")
    # PRD 12: a kernel has no .git, so stage the code's commit for breachlab.train.repro
    (dest / "GIT_SHA").write_text(git_sha(root=ROOT) + "\n", encoding="utf-8")
    (dest / "dataset-metadata.json").write_text(
        json.dumps(dataset_meta(OWNER, CODE_SLUG, "breachlab code")), encoding="utf-8"
    )


def stage_data(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in (
        "train",
        "val",
        "mining_pool",
        "gold_test",
    ):  # gold_test is only read by eval stages
        shutil.copy(src / f"{name}.jsonl", dest / f"{name}.jsonl")
    (dest / "dataset-metadata.json").write_text(
        json.dumps(
            dataset_meta(OWNER, DATA_SLUG, f"breachlab {DATA_SLUG.removeprefix('breachlab-')}")
        ),
        encoding="utf-8",
    )


def kaggle(*args: str) -> str:  # pragma: no cover - network
    out = subprocess.run(
        [sys.executable, "-m", "kaggle", *args], capture_output=True, text=True, check=False
    )
    if out.returncode:
        raise RuntimeError((out.stdout + out.stderr)[-800:])
    return out.stdout


def upload_dataset(folder: Path, slug: str) -> None:  # pragma: no cover - network
    try:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update", "--dir-mode", "zip")
    except RuntimeError:
        kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip")


def push_kernel(
    slug: str,
    model: str,
    seed: int,
    workdir: Path,
    extra: str = "",
    kernels: list[str] | None = None,
) -> str:  # pragma: no cover - network
    workdir.mkdir(parents=True, exist_ok=True)
    ds = [f"{OWNER}/{CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"]
    (workdir / "kernel-metadata.json").write_text(
        json.dumps(
            kernel_meta(
                OWNER, slug, ds, gpu=not slug.startswith("breachlab-export"), kernels=kernels
            )
        ),
        encoding="utf-8",
    )
    (workdir / "run.py").write_text(
        kernel_script(model, seed, f"/kaggle/input/{DATA_SLUG}", extra), encoding="utf-8"
    )
    return kaggle("kernels", "push", "-p", str(workdir))


if __name__ == "__main__":  # pragma: no cover - network
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", type=Path, help="dir with train/val/mining_pool.jsonl (uploads data+code)"
    )
    ap.add_argument("--slug", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="train")
    ap.add_argument("--from-kernel", default="", help="owner/slug whose output holds the adapters")
    ap.add_argument("--no-dpo", action="store_true", help="SFT only (the shipped recipe)")
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    stage_code(work / "code")
    upload_dataset(work / "code", CODE_SLUG)
    if a.data:
        stage_data(a.data, work / "data")
        upload_dataset(work / "data", DATA_SLUG)
    print(
        push_kernel(
            a.slug,
            a.model,
            a.seed,
            work / a.slug,
            f", '--stage', {a.stage!r}" + (", '--no-dpo'" if a.no_dpo else ""),
            [a.from_kernel] if a.from_kernel else None,
        )
    )
