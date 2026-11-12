"""Push the public-safe splits + dataset card to an HF dataset repo (private until Phase 7, ADR-0014).

Only public-source and synthetic rows exist in v1 (no live data), so every split is public-safe.
Usage: uv run python scripts/publish_data.py data/processed/v1 giri-harsh/breachlab-data
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main(d: Path, repo: str) -> int:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for name in ("train", "val", "mining_pool", "gold_test"):
        api.upload_file(
            path_or_fileobj=str(d / f"{name}.jsonl"),
            path_in_repo=f"{name}.jsonl",
            repo_id=repo,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=str(d / "DATASET_CARD.md"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print("pushed", repo)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2]))
