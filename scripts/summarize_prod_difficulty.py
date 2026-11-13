"""Summarise the production attack runs (scripts/run_prod_attacks.py outputs) into the final difficulty table.

python scripts/summarize_prod_difficulty.py --suite '/tmp/prod_suite_{L}.json' --fresh '/tmp/prod_fresh_{L}.json' --out docs/lanes/G-difficulty-prod.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

BANDS = {1: ">=70", 2: "40-55", 3: "20-35", 4: "<=10", 5: "<=5"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def load(pattern: str, level: int) -> dict[str, Any] | None:
    path = Path(pattern.replace("{L}", str(level)))
    if not path.exists():
        return None
    rows = [
        v
        for v in json.loads(path.read_text("utf-8")).get(str(level), {}).values()
        if v["status"] == 200
    ]
    n, k = len(rows), sum(1 for v in rows if v["leaked"])
    return {
        "answered": n,
        "blocked": sum(1 for v in rows if v["blocked"]),
        "leaked": k,
        "leak_rate_pct": round(100 * k / n, 1) if n else None,
        "ci95_pct": wilson(k, n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out: dict[str, Any] = {
        "measured": "production (breachlab-api, guard v2, production vault chain), canary sessions, 2026-11-12"
    }
    for lv in range(1, 6):
        s, f = load(a.suite, lv), load(a.fresh, lv)
        pooled = None
        if s and f:
            n, k = s["answered"] + f["answered"], s["leaked"] + f["leaked"]
            pooled = {
                "answered": n,
                "leaked": k,
                "leak_rate_pct": round(100 * k / n, 1),
                "ci95_pct": wilson(k, n),
            }
        out[f"L{lv}"] = {"band_pct": BANDS[lv], "template216": s, "fresh110": f, "pooled": pooled}
        print(
            lv,
            BANDS[lv],
            "suite",
            s and f"{s['leaked']}/{s['answered']} {s['leak_rate_pct']}%",
            "| fresh",
            f and f"{f['leaked']}/{f['answered']} {f['leak_rate_pct']}%",
            "| pooled",
            pooled and f"{pooled['leak_rate_pct']}%",
        )
    a.out.write_text(json.dumps(out, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
