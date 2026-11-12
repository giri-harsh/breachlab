"""After the Modal build: verify PRD 8.2 targets, log stats to MLflow, write the gold review sample.

Usage: modal volume get breachlab-data v1 data/processed/ && uv run python scripts/finalize_data.py data/processed/v1
Prints the target report; exits 1 on shortfall so the caller can run a top-up job.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from breachlab.data.card import stats
from breachlab.data.verify import verify_targets
from breachlab.train.scoring import read_rows


def main(d: Path, version: str = "v1") -> int:
    rows = [
        r for s in ("train", "val", "mining_pool", "gold_test") for r in read_rows(d / f"{s}.jsonl")
    ]
    rep = verify_targets(rows)
    print(json.dumps(rep.__dict__, indent=2))
    gold = [r for r in rows if r.split == "gold_test"]
    sample = random.Random(7).sample(gold, min(300, len(gold)))
    (d / "review_queue_gold300.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in sample) + "\n", encoding="utf-8"
    )
    try:
        import mlflow

        mlflow.set_experiment("breachlab-data")
        with mlflow.start_run(run_name=f"data-stats-{version}"):
            mlflow.log_metrics(
                {
                    "train_rows": rep.train_rows,
                    "unsafe_share": rep.unsafe_share,
                    "hard_benign_share_of_safe": rep.hard_benign_share_of_safe,
                    "hi_hinglish_share_of_unsafe": rep.hi_hinglish_share_of_unsafe,
                }
            )
            mlflow.log_dict(stats(rows), "stats.json")
            mlflow.log_param("targets_ok", rep.ok)
    except Exception as exc:  # noqa: BLE001 - tracking must not block the data decision
        print("mlflow logging failed:", type(exc).__name__, file=sys.stderr)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
