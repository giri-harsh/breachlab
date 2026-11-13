"""Make the BreachLab Hugging Face repos public after checking licences and the GitHub link in their cards (H2/H4).

    python scripts/hf_publish.py            # check only
    python scripts/hf_publish.py --apply    # check, then flip private repos to public

Refuses to publish when the licence tag or the code-repository link is missing. Needs HF_TOKEN; the token is never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPOS = [
    ("Pseudo-Harsh/breachlab-guard-v1", "model", "license:apache-2.0"),
    ("Pseudo-Harsh/breachlab-data", "dataset", "license:cc-by-sa-3.0"),
]
CODE_REPO = "https://github.com/giri-harsh/breachlab"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    api = HfApi(token=os.environ["HF_TOKEN"])
    ok = True
    for rid, rtype, lic in REPOS:
        info = api.repo_info(rid, repo_type=rtype)
        card = api.hf_hub_download(rid, "README.md", repo_type=rtype)
        text = Path(card).read_text(encoding="utf-8")
        good = lic in (info.tags or []) and CODE_REPO in text
        print(
            f"{rid}: private={info.private} licence tag {lic}: {lic in (info.tags or [])} github link in card: {CODE_REPO in text}"
        )
        ok &= good
        if a.apply and good and info.private:
            api.update_repo_settings(rid, private=False, repo_type=rtype)
            print("  -> now public")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
