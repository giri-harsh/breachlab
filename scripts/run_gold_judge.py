"""Gemini judge pass over all gold_test rows: label double-check (PRD 8.3) + the 'gemini-judge' comparison row.

    uv run python scripts/run_gold_judge.py [--rounds 12] [--wait 300]

Resumable: judgments are cached in data/eval/gold_judge.json (id -> label/category/p_unsafe/model). Writes
data/eval/gold_judge_stats.json (agreement, kappa, slices), data/eval/gold_judge_disagreements.json, and, once every
row is judged, the `gemini-judge` entry of data/eval/api_preds.json. API only, no weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from breachlab.data.llm import GeminiPool, LLMClient
from breachlab.eval import gold_judge as gj
from breachlab.train.scoring import read_rows

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data" / "eval"
MODELS = ("gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-3.5-flash-lite")


def clients() -> list[tuple[str, LLMClient]]:  # pragma: no cover - network
    return [(m, GeminiPool(models=(m,))) for m in MODELS]


def finish(rows, cache) -> None:  # type: ignore[no-untyped-def]
    stats = gj.agreement_stats(rows, cache)
    (EV / "gold_judge_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    dis = gj.disagreements(rows, cache)
    (EV / "gold_judge_disagreements.json").write_text(json.dumps(dis, indent=1), encoding="utf-8")
    preds = gj.judge_row_preds(rows, cache)
    if preds is not None:
        ap = EV / "api_preds.json"
        api = json.loads(ap.read_text()) if ap.exists() else {}
        api["gemini-judge"] = preds
        ap.write_text(json.dumps(api), encoding="utf-8")
    print(
        {k: stats[k] for k in ("n_judged", "n_missing", "agreement", "kappa")},
        "disagreements:",
        len(dis),
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "data/processed/v1/gold_test.jsonl")
    ap.add_argument("--rounds", type=int, default=12, help="retry rounds when quota is exhausted")
    ap.add_argument("--wait", type=float, default=300.0, help="seconds between rounds")
    a = ap.parse_args(argv)
    rows = read_rows(a.gold)
    path = EV / "gold_judge.json"
    cache = gj.load_cache(path)
    for rnd in range(a.rounds):
        # fresh pools every round: a pool escalates its 429 cooldown up to 1 h, which would outlast a quota reset
        gj.run_judge(rows, cache, clients(), on_progress=lambda c: gj.save_cache(path, c))
        gj.save_cache(path, cache)
        missing = sum(r.id not in cache for r in rows)
        print(f"round {rnd}: judged {len(cache)}/{len(rows)}", flush=True)
        if not missing:
            break
        time.sleep(a.wait)
    finish(rows, cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
