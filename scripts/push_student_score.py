"""Push the CPU kernel that scores splits with the served int8 students (see breachlab.serve.student_job)."""

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

PIP = ["pydantic-settings", "pyyaml", "pandera", "scipy", "onnxruntime", "tokenizers", "numpy"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--source", default="giriharsh/breachlab-distill-v1")
    ap.add_argument("--candidates", default="distilbert-ml,minilm-ml")
    ap.add_argument("--splits", default="val,gold_test,opener_slice_v1,ood_test")
    ap.add_argument("--upload-code", action="store_true")
    a = ap.parse_args()
    work = ROOT / "kaggle" / "_work"
    if a.upload_code:
        import time

        stage_code(work / "code")
        upload_dataset(work / "code", CODE_SLUG)
        while "ready" not in kaggle("datasets", "status", f"{OWNER}/{CODE_SLUG}"):
            time.sleep(10)
        time.sleep(60)
    d = work / a.slug
    d.mkdir(parents=True, exist_ok=True)
    meta = kernel_meta(
        OWNER,
        a.slug,
        [f"{OWNER}/{CODE_SLUG}", f"{OWNER}/{DATA_SLUG}"],
        gpu=False,
        kernels=[a.source],
    )
    (d / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    script = (
        "import glob, os, subprocess, sys\n"
        f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *{PIP!r}])\n"
        "pkg = glob.glob('/kaggle/input/**/src/breachlab', recursive=True)[0]\n"
        "root = os.path.dirname(os.path.dirname(pkg))\n"
        "os.environ['BREACHLAB_CONFIG_DIR'] = os.path.join(root, 'configs')\n"
        "sys.path.insert(0, os.path.join(root, 'src'))\n"
        "data = os.path.dirname(glob.glob('/kaggle/input/**/gold_test.jsonl', recursive=True)[0])\n"
        "from breachlab.serve.student_job import main\n"
        f"sys.exit(main(['--data-dir', data, '--out', '/kaggle/working/out', '--candidates', {a.candidates!r}, '--splits', {a.splits!r}]))\n"
    )
    (d / "run.py").write_text(script, encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
