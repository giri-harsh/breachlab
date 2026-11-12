"""Finish a data build in the dev container from cached seeds + shards (CPU only, no weights).

Used when the Modal workspace cannot start new jobs (spend limit): the container only does dedup / splits /
validation plus a few hundred hard-benign LLM API calls. Usage: python scripts/build_local.py data/processed/v1
"""

from __future__ import annotations

import sys
from pathlib import Path

from breachlab.data.build import build
from breachlab.data.llm import GeminiPool, GroqClient, LockedRoundRobin
from breachlab.data.schema import Row, jsonl_lines


def main(out: Path, max_llm_shards: int = 32) -> int:
    seeds = [
        Row.model_validate_json(x) for x in jsonl_lines((out / "seeds.jsonl").read_text("utf-8"))
    ]
    hb = LockedRoundRobin(
        [GroqClient(retries=2), GeminiPool(), GroqClient("openai/gpt-oss-20b", retries=2)],
        rounds=40,
        pause=30,
    )
    rows = build(
        seeds,
        hb,  # never called for mutation: every shard is cached
        out,
        batch=True,
        k_det=2,
        workers=6,
        hb_llm_calls=0,
        n_combinatorial=2600,
        n_quoted=3000,
        extra_det=2,
        hb_llm=hb,
        balance=True,
        max_llm_shards=max_llm_shards,
    )
    print("rows", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
