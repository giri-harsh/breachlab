"""Experiment-tracking interface; MLflow implementation plus a no-network fake for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifact: Path | None = None,
    ) -> str: ...


class MlflowTracker:
    def __init__(self, experiment: str = "breachlab-phase1") -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(experiment)

    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifact: Path | None = None,
    ) -> str:
        with self._mlflow.start_run(run_name=name) as run:
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
            if artifact is not None:
                self._mlflow.log_artifact(str(artifact))
            return str(run.info.run_id)


class FakeTracker:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, str], dict[str, float], Path | None]] = []

    def log_run(
        self,
        name: str,
        params: dict[str, str],
        metrics: dict[str, float],
        artifact: Path | None = None,
    ) -> str:
        self.runs.append((name, params, metrics, artifact))
        return f"fake-{len(self.runs)}"
