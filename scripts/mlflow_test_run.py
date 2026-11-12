"""Phase 1 acceptance: log a param, a metric and an artifact to the MLflow tracking server."""

import os
import sys
from pathlib import Path

from breachlab.common.tracking import MlflowTracker


def main() -> int:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLFLOW_TRACKING_URI not set", file=sys.stderr)
        return 1
    art = Path("reports/mlflow-test-artifact.txt")
    art.parent.mkdir(exist_ok=True)
    art.write_text("breachlab phase1 smoke\n", encoding="utf-8")
    run_id = MlflowTracker().log_run("phase1-smoke", {"phase": "1"}, {"ok": 1.0}, art)
    print(f"mlflow run id: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
