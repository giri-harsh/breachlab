"""Push one Kaggle kernel for an existing pipeline stage WITHOUT re-uploading the shared code/data datasets.

    uv run python scripts/run_kaggle_stage.py --slug breachlab-evalgold-06b-s2 --stage evalgold --seed 2 \
        --from-kernel giriharsh/breachlab-sft-06b-s2

`breachlab.train.kaggle_job`'s CLI always versions the `breachlab-code` dataset from the caller's checkout, which
would silently overwrite another lane's newer copy. This wrapper reuses its kernel/script builders and pushes only
the kernel; the datasets already on Kaggle are used as they are.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from breachlab.train import kaggle_job

STAGES = ("train", "evalgold", "export", "zeroshot", "deberta")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--stage", choices=STAGES, default="train")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-kernel", default="")
    a = ap.parse_args(argv)
    wd = Path(tempfile.mkdtemp(prefix=a.slug))
    print(
        kaggle_job.push_kernel(
            a.slug,
            a.model,
            a.seed,
            wd,
            f", '--stage', {a.stage!r}",
            [a.from_kernel] if a.from_kernel else None,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
