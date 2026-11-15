"""Relay the int8 student from the distillation kernel output to the Hub (transit only, ADR-0040) as ``cascade-v1/``.

    python scripts/relay_student.py minilm-ml /path/student-meta.json [--dest cascade-v1] [--kernel giriharsh/breachlab-distill-v1]

Uploads ``student-int8.onnx``, ``student-tokenizer.json`` (the student's tokenizer.json) and ``student-meta.json``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Pseudo-Harsh/breachlab-guard-v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("meta", type=Path)
    ap.add_argument("--dest", default="cascade-v1")
    ap.add_argument("--kernel", default="giriharsh/breachlab-distill-v1")
    a = ap.parse_args()
    exe = str(Path(sys.executable).with_name("kaggle"))
    api = HfApi(token=os.environ["HF_TOKEN"])
    tmp = Path(tempfile.mkdtemp(prefix="student-relay-"))
    try:
        pat = rf".*out/{a.candidate}/(student-int8\.onnx|tokenizer\.json)$"
        r = subprocess.run(
            [exe, "kernels", "output", a.kernel, "-p", str(tmp), "--file-pattern", pat],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode:
            raise SystemExit((r.stdout + r.stderr)[-400:])
        onnx = next(tmp.rglob("student-int8.onnx"))
        tok = next(tmp.rglob("tokenizer.json"))
        for src, name in (
            (onnx, "student-int8.onnx"),
            (tok, "student-tokenizer.json"),
            (a.meta, "student-meta.json"),
        ):
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"{a.dest}/{name}",
                repo_id=REPO,
                repo_type="model",
            )
            print("uploaded", f"{a.dest}/{name}", src.stat().st_size, "bytes")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
