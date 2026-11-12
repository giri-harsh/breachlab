"""Pre-public scan of the git-tracked tree (H1): secrets, risky files, AI attribution. Prints findings, exits 1 if any.

    python scripts/prepublic_scan.py [--tree DIR]

Exact environment secret values (never printed), token patterns, private-key headers, database URLs with passwords,
dataset/weight files, and AI attribution strings in file contents. Values are reported as counts and file names only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|URL|DSN", re.I)
PATTERNS = {
    "hf token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "groq key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    "neon password": re.compile(r"npg_[A-Za-z0-9]{8,}"),
    "kaggle token": re.compile(r"KGAT_[A-Za-z0-9]{10,}"),
    "google key": re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "db url with password": re.compile(r"postgres(?:ql)?://[^\s:@/]+:[^\s@/]{6,}@"),
    "grafana/langfuse": re.compile(
        r"glc_[A-Za-z0-9]{20,}|sk-lf-[0-9a-f-]{20,}|pk-lf-[0-9a-f-]{20,}"
    ),
    "vercel token": re.compile(r"vcp_[A-Za-z0-9]{20,}"),
}
ATTRIBUTION = re.compile(
    r"co-authored-by:.*(claude|anthropic)|claude-session:|generated (with|by) \[?claude|noreply@anthropic\.com|claude\.ai/code/session",
    re.I,
)
RISKY_EXT = {
    ".jsonl",
    ".npy",
    ".npz",
    ".onnx",
    ".safetensors",
    ".bin",
    ".pt",
    ".parquet",
    ".pem",
    ".key",
}
ALLOWED_ATTRIBUTION_FILES = {
    "scripts/check_no_ai_trailers.py",
    "tests/scripts/test_scripts.py",
    "scripts/prepublic_scan.py",
}  # the hook, its test and this scanner must contain the patterns
ALLOWED_URL_FIXTURES = {"tests/monitor/test_workflows.py"}  # a dummy postgres URL fixture
ALLOWED_DATA = {
    "data/review/opener_slice_v1.jsonl",
    "docs/lanes/fresh_attacks_v1.jsonl",
    "docs/lanes/fresh_attacks_v2.jsonl",
}


def tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    )
    return [p for p in out.stdout.decode().split("\0") if p]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, default=Path("."))
    a = ap.parse_args(argv)
    root = a.tree
    secrets = [
        v
        for k, v in os.environ.items()
        if SECRET_ENV.search(k) and len(v) >= 12 and "://localhost" not in v
    ]
    findings: list[str] = []
    files = tracked(root)
    for rel in files:
        path = root / rel
        if Path(rel).suffix in RISKY_EXT and rel not in ALLOWED_DATA:
            findings.append(f"risky file type: {rel}")
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        for v in secrets:
            if v in text:
                findings.append(f"exact secret value (one of {len(secrets)} env values) in {rel}")
                break
        for name, rx in PATTERNS.items():
            if name == "db url with password" and rel in ALLOWED_URL_FIXTURES:
                continue
            if rx.search(text):
                findings.append(f"{name} pattern in {rel}")
        if rel not in ALLOWED_ATTRIBUTION_FILES and ATTRIBUTION.search(text):
            findings.append(f"AI attribution string in {rel}")
    print(f"files scanned: {len(files)}, env values checked: {len(secrets)}")
    for f in sorted(set(findings)):
        print("FINDING", f)
    print("VERDICT", "SAFE" if not findings else "NOT SAFE")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
