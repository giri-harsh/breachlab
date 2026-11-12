"""README headline numbers come only from web/public/scorecard.json (itself written from reports/eval-vN.md).

Markers: ``<!--num:MODEL.METRIC-->value<!--/num-->`` with METRIC in asr, fpr, f1, auroc, p95_ms.
  python scripts/readme_numbers.py fill    # rewrite the values in README.md
  python scripts/readme_numbers.py check   # exit 1 if README disagrees with the scorecard (CI)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = re.compile(r"<!--num:([^<>]+)\.(\w+)-->(.*?)<!--/num-->")


def fmt(metric: str, v: float | None) -> str:
    if v is None:
        return "n/a"
    return (
        f"{v:.0f} ms"
        if metric == "p95_ms"
        else f"{v:.3f}"
        if metric in ("f1", "auroc")
        else f"{v * 100:.1f}%"
    )


def expected(card: dict, model: str, metric: str) -> str:  # type: ignore[type-arg]
    return fmt(metric, card["models"][model].get(metric))


def process(text: str, card: dict, write: bool) -> tuple[str, list[str]]:  # type: ignore[type-arg]
    problems: list[str] = []

    def sub(m: re.Match[str]) -> str:
        model, metric, cur = m.groups()
        want = expected(card, model, metric)
        if cur != want:
            problems.append(f"{model}.{metric}: README has {cur!r}, scorecard says {want!r}")
        return f"<!--num:{model}.{metric}-->{want if write else cur}<!--/num-->"

    return MARK.sub(sub, text), problems


def main(mode: str) -> int:
    card = json.loads((ROOT / "web/public/scorecard.json").read_text())
    path = ROOT / "README.md"
    new, problems = process(path.read_text(encoding="utf-8"), card, mode == "fill")
    if mode == "fill":
        path.write_text(new, encoding="utf-8")
        return 0
    for p in problems:
        print("MISMATCH", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main("check"))
