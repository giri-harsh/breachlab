"""DATASET_CARD.md generator: counts by source/family/language/label/split."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from breachlab.data.schema import Row

DIMENSIONS: list[tuple[str, Callable[[Row], str]]] = [
    ("label", lambda r: r.label),
    ("split", lambda r: r.split),
    ("source", lambda r: r.source),
    ("attack_family", lambda r: r.attack_family),
    ("language", lambda r: r.language),
    ("encoding", lambda r: r.encoding),
]


def stats(rows: list[Row]) -> dict[str, dict[str, int]]:
    return {name: dict(Counter(fn(r) for r in rows)) for name, fn in DIMENSIONS}


def render_card(rows: list[Row], version: str, licenses: dict[str, str] | None = None) -> str:
    out = [f"# BreachLab dataset {version}", "", f"Total rows: {len(rows)}", ""]
    for name, counts in stats(rows).items():
        out += [f"## By {name}", "", "| value | count |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
        out.append("")
    if licenses:
        out += ["## Source licenses", "", "| source | license |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in sorted(licenses.items())]
        out.append("")
    out.append("Research demo. Live game data is never included in the public dataset.")
    return "\n".join(out) + "\n"
