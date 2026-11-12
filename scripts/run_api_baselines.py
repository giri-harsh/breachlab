"""API-backed comparison rows on gold_test: regex and Groq Prompt Guard 2 (the Gemini judge row: run_gold_judge.py).

Writes data/eval/api_preds.json: {model: {id: [label, category, p_unsafe]}}. Resumable (skips finished
models); CPU/API only, no weights. Usage: python scripts/run_api_baselines.py data/processed/v1/gold_test.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from breachlab.eval.external import prompt_guard_scores
from breachlab.serve.baselines import regex_classify
from breachlab.train.scoring import read_rows

OUT = Path("data/eval/api_preds.json")


def main(gold: Path) -> int:
    rows = read_rows(gold)
    texts = [r.text for r in rows]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict[str, list[object]]] = json.loads(OUT.read_text()) if OUT.exists() else {}

    def save() -> None:
        OUT.write_text(json.dumps(res), encoding="utf-8")

    if "regex" not in res:
        res["regex"] = {
            r.id: [*regex_classify(r.text), 1.0 if regex_classify(r.text)[0] == "UNSAFE" else 0.0]
            for r in rows
        }
        save()
    if "llama-prompt-guard-2-86m" not in res:
        scores = prompt_guard_scores(texts, os.environ["GROQ_API_KEY"], workers=2)
        res["llama-prompt-guard-2-86m"] = {
            r.id: ["UNSAFE" if s >= 0.5 else "SAFE", "none", s]
            for r, s in zip(rows, scores, strict=True)
        }
        save()
    if "gemini-judge" not in res:
        # judge_batch used to turn unparsed/failed chunks into silent SAFE verdicts; the judge row now comes from
        # scripts/run_gold_judge.py (cached, strict parsing, also gives the gold label double-check).
        print("gemini-judge row: run scripts/run_gold_judge.py")
    print({k: len(v) for k, v in res.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
