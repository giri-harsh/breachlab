"""Register trained guard versions (metadata only, no weights) in the MLflow registry with aliases.

Each version points at its Kaggle kernel and carries val/gold metrics; weights live in the kernel output and
(once Modal is available) on HF. Usage: python scripts/register_runs.py
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "data" / "eval"


def main() -> int:
    mlflow.set_experiment("breachlab-train")
    client = mlflow.MlflowClient()
    rep = json.loads((ROOT / "web/public/scorecard.json").read_text())["models"]
    res = json.loads((EV / "dpo_results.json").read_text())
    specs = [
        (
            "guard-v1-0.6b-sft-s0",
            "qwen3-0.6b-sft",
            res["sft"]["calibration"],
            "giriharsh/breachlab-sft-06b-s0",
            6229.0,
        ),
        (
            "guard-v1-0.6b-sft-dpo-s0",
            "qwen3-0.6b-finetuned",
            res["dpo"]["calibration"],
            "giriharsh/breachlab-dpo-06b-s0",
            6448.0,
        ),
    ]
    versions = []
    for name, key, cal, kernel, secs in specs:
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {"kernel": kernel, "base_model": "Qwen/Qwen3-0.6B", "data": "data-v1", "seed": 0}
            )
            mlflow.log_metrics(
                {f"gold_{k}": float(v) for k, v in rep[key].items() if isinstance(v, int | float)}
            )
            mlflow.log_metrics(
                {
                    "gpu_seconds": secs,
                    "gpu_usd": 0.0,
                    "temperature": cal["temperature"],
                    "threshold": cal["threshold"],
                    "ece_after": cal["ece_after"],
                }
            )
            mlflow.log_dict({"name": name, "kernel": kernel, "calibration": cal}, "results.json")
        with contextlib.suppress(mlflow.exceptions.MlflowException):  # already exists
            client.create_registered_model("breachlab-guard")
        mv = client.create_model_version(
            "breachlab-guard",
            f"runs:/{run.info.run_id}/results.json",
            run.info.run_id,
            tags={"kernel": kernel},
        )
        versions.append(mv.version)
        print(name, "-> version", mv.version, "run", run.info.run_id)
    client.set_registered_model_alias("breachlab-guard", "challenger", versions[-1])
    print("challenger =", versions[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
