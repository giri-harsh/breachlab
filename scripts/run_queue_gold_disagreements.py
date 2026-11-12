"""Append gold rows where the Gemini judge disagrees with the generation-time label to the review session.

    uv run python scripts/run_queue_gold_disagreements.py [data/eval/gold_judge_disagreements.json]

Same HTTPS-SQL path as scripts/queue_gold_review.py (the sandbox cannot open Postgres TCP). Rows go into the existing
session `gold-review-v1` at level 0 (verdict NULL, confidence 0.5). Idempotent: messages already present in that
session are skipped. Requires NEON_DATABASE_URL; the URL is never printed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from breachlab.eval.gold_judge import review_queries

SID = "gold-review-v1"
BATCH = 120  # statements per HTTPS call (attempt + label = 2 statements per row)


def post(ep: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    r = httpx.post(ep, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return dict(r.json())


def main(path: Path) -> int:
    url = os.environ["NEON_DATABASE_URL"]
    ep = f"https://{urlsplit(url).hostname}/sql"
    headers = {"Neon-Connection-String": url, "Content-Type": "application/json"}
    items = json.loads(path.read_text("utf-8"))
    sess = post(
        ep, headers, {"query": "select count(*) as n from sessions where id=$1", "params": [SID]}
    )
    if int(sess["rows"][0]["n"]) == 0:
        print(f"session {SID} missing: run scripts/queue_gold_review.py first")
        return 1
    got = post(
        ep, headers, {"query": "select message from attempts where session_id=$1", "params": [SID]}
    )
    existing = {str(r["message"]) for r in got["rows"]}
    qs = review_queries(SID, items, existing)
    for i in range(0, len(qs), BATCH):
        post(
            ep, headers, {"queries": qs[i : i + BATCH]}
        )  # BATCH is even: attempt + label stay together
    print(
        f"queued {len(qs) // 2} of {len(items)} disagreements ({len(items) - len(qs) // 2} already present)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(
        main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/gold_judge_disagreements.json"))
    )
