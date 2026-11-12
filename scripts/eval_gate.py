"""CI / promote gate (PRD 9.5). Input JSON: {"candidate": CandidateResult, "champion": ChampionResult}.

Exits 1 when any check fails. A missing latency measurement fails closed (p95 must be measured on CPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from breachlab.eval.gate import CandidateResult, ChampionResult, evaluate_gate


def main(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cand_raw = dict(raw["candidate"])
    if cand_raw.get("p95_latency_ms") is None:
        print("FAIL latency: not measured")
        return 1
    out = evaluate_gate(CandidateResult(**cand_raw), ChampionResult(**raw["champion"]))
    for name, (ok, msg) in out.checks.items():
        print(("PASS" if ok else "FAIL"), name, msg)
    print("GATE", "PASSED" if out.passed else "FAILED")
    return 0 if out.passed else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
