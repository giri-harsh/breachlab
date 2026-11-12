"""Queue the 300-row gold_test sample for /admin/review (PRD 8.3, non-blocking for v1).

Writes one non-canary 'gold-review' session and 300 attempts (level 0, message = gold text, verdict NULL,
confidence 0.5 so they sort to the top of the queue) plus a judge-source label holding the generation-time label.
Uses Neon's HTTPS SQL endpoint because the sandbox cannot open Postgres TCP. Idempotent via a fixed session id.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

SID = "gold-review-v1"


def main(queue: Path) -> int:
    url = os.environ["NEON_DATABASE_URL"]
    host = urlsplit(url).hostname
    rows = [json.loads(x) for x in queue.read_text("utf-8").split("\n") if x.strip()]
    h = {"Neon-Connection-String": url, "Content-Type": "application/json"}
    ep = f"https://{host}/sql"
    r = httpx.post(
        ep,
        headers=h,
        json={"query": "select count(*) as n from attempts where session_id=$1", "params": [SID]},
        timeout=60,
    )
    r.raise_for_status()
    if int(r.json()["rows"][0]["n"]) >= len(rows):
        print("already queued")
        return 0
    qs: list[dict[str, object]] = [
        {"query": "insert into players (nickname) values ($1)", "params": ["gold-review"]},
        {
            "query": "insert into sessions (id, player_id, ip_hash, is_canary) values ($1, currval(pg_get_serial_sequence('players','id')), $2, false) on conflict do nothing",
            "params": [SID, "gold"],
        },
    ]
    for x in rows:
        qs.append(
            {
                "query": "insert into attempts (session_id, level, message, confidence, leaked) values ($1, 0, $2, 0.5, false)",
                "params": [SID, x["text"]],
            }
        )
        qs.append(
            {
                "query": "insert into labels (attempt_id, label, category, source) values (currval(pg_get_serial_sequence('attempts','id')), $1, $2, 'judge')",
                "params": [x["label"], x["category"] if x["label"] == "UNSAFE" else None],
            }
        )
    r = httpx.post(ep, headers=h, json={"queries": qs}, timeout=120)
    r.raise_for_status()
    print("queued", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
