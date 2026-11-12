"""Register guard v2 (metadata only, no weights) in the MLflow registry, run the gate, promote it to champion (D4).

    python scripts/register_v2.py

Numbers come from docs/lanes/T-guard-v2-gates.json (held-out opener set, gold_test) and web/public/scorecard.json.
The gate is evaluated with the fp32-reference convention of reports/gate-input.json, using the v2 served-int4 quality;
the champion alias moves with ``breachlab.flywheel.registry.promote`` (previous = v1, rollback = registry rollback).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow

from breachlab.eval.gate import CandidateResult, ChampionResult, evaluate_gate
from breachlab.flywheel.registry import MODEL, promote

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    gates = json.loads((ROOT / "docs/lanes/T-guard-v2-gates.json").read_text("utf-8"))
    card = json.loads((ROOT / "web/public/scorecard.json").read_text("utf-8"))["models"]
    v2, v1 = card["qwen3-0.6b-sft-int4-v2"], card["qwen3-0.6b-sft-int4"]
    cal = json.loads((ROOT / "docs/lanes/T-v2-calibration-q4b32.json").read_text("utf-8"))
    cand = CandidateResult(
        asr=v2["asr"],
        fpr=v2["fpr"],
        p95_latency_ms=v2["p95_ms"],
        ece=cal["ece_after"],
        mcnemar_p=None,
    )
    out = evaluate_gate(cand, ChampionResult(asr=v1["asr"], fpr=v1["fpr"]))
    for name, (ok, msg) in out.checks.items():
        print("PASS" if ok else "FAIL", name, msg)
    mlflow.set_experiment("breachlab-train")
    client = mlflow.MlflowClient()
    with mlflow.start_run(run_name="guard-v2-0.6b-sft-s0") as run:
        mlflow.log_params(
            {
                "kernel": "giriharsh/breachlab-sft-06b-v2-s0",
                "base_model": "Qwen/Qwen3-0.6B",
                "data": "data-v2",
                "seed": 0,
                "hub_folder": "champion-v2",
            }
        )
        mlflow.log_metrics(
            {f"gold_{k}": float(v) for k, v in v2.items() if isinstance(v, int | float)}
        )
        mlflow.log_metrics(
            {
                "opener_fpr": gates["openers"]["overall"]["rate"],
                "hindi_opener_fpr": gates["openers"]["by_language"]["hi"]["rate"],
                "temperature": cal["temperature"],
                "threshold": cal["threshold"],
                "gpu_seconds": 6417.2,
                "gpu_usd": 0.0,
            }
        )
        mlflow.log_dict({"gates": gates["pass"], "calibration": cal}, "results.json")
    mv = client.create_model_version(
        MODEL,
        f"runs:/{run.info.run_id}/results.json",
        run.info.run_id,
        tags={"kernel": "giriharsh/breachlab-sft-06b-v2-s0", "hub_folder": "champion-v2"},
    )
    client.set_registered_model_alias(MODEL, "challenger", mv.version)
    print("registered version", mv.version, "gates", gates["pass"])
    if not gates["all_pass"]:
        print("A4 gates not all passed: stays challenger")
        return 1
    move = promote(client, mv.version)
    print("promoted:", move)
    return 0


if __name__ == "__main__":
    sys.exit(main())
