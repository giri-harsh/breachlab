"""Push the distillation Kaggle kernel (teacher soft labels + students + int8 export). Uses the v2 data dataset.

BREACHLAB_DATA_SLUG=breachlab-data-v2 python scripts/push_distill.py --slug breachlab-distill-v1 [--candidates ...]
"""

from __future__ import annotations

import argparse
import json

from breachlab.train.kaggle_job import (
    CODE_SLUG,
    DATA_SLUG,
    OWNER,
    ROOT,
    kaggle,
    kernel_meta,
    stage_code,
    upload_dataset,
)

PIP = [
    "transformers>=4.51",
    "torchao>=0.16.0",
    "peft",
    "accelerate",
    "pydantic-settings",
    "pyyaml",
    "pandera",
    "scipy",
    "scikit-learn",
    "sentencepiece",
    "protobuf",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "tokenizers",
]


def script(extra: str) -> str:
    return (
        "import glob, os, subprocess, sys\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "data = os.path.dirname(glob.glob('/kaggle/input/**/train.jsonl', recursive=True)[0])\n"
        "from breachlab.train.distill_job import main\n"
        f"sys.exit(main(['--data-dir', data, '--out', '/kaggle/working/out'{extra}]))\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--teacher-kernel", default="giriharsh/breachlab-sft-06b-v2-s0")
    ap.add_argument("--candidates", default="minilm-ml,distilbert-ml,xlmr-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--extra", default="", help="extra kernel args, python list fragment")
    ap.add_argument("--skip-code-upload", action="store_true")
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    if not a.skip_code_upload:
        stage_code(work / "code")
        upload_dataset(work / "code", CODE_SLUG)
        import time

        while "ready" not in kaggle("datasets", "status", f"{OWNER}/{CODE_SLUG}"):
            time.sleep(10)
        time.sleep(60)
    d = work / a.slug
    d.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER,
        a.slug,
        [f"{OWNER}/{CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"],
        gpu=True,
        kernels=[a.teacher_kernel],
    )
    (d / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    extra = f", '--candidates', {a.candidates!r}, '--epochs', '{a.epochs}'{a.extra}"
    (d / "run.py").write_text(script(extra), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
