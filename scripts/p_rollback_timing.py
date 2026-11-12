"""Rollback timing (PRD 10.8: takes effect < 1 min; test once and record timing) on a SEPARATE test model.

    uv run python scripts/p_rollback_timing.py [--model breachlab-guard-rollback-test] [--reps 3]
        [--out docs/lanes/P-rollback-timing.json]

Uses ``breachlab.flywheel.registry.promote`` / ``rollback`` against the MLflow registry in MLFLOW_TRACKING_URI
(DagsHub; credentials from MLFLOW_TRACKING_USERNAME / PASSWORD, never printed). It creates (once) a test registered
model with two tiny dummy versions that point at a small JSON artifact (no weights), sets ``champion`` to v1, then
times ``--reps`` cycles of promote(v2) and rollback, each followed by a read-back of the ``champion`` alias. It
refuses to touch the production model ``breachlab-guard``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breachlab.flywheel.registry import MODEL, RegistryClient, promote, rollback  # noqa: E402

TEST_MODEL = "breachlab-guard-rollback-test"


def check_name(name: str) -> str:
    if name == MODEL or not name.endswith("-test"):
        raise ValueError(
            f"refusing to time on {name!r}: use a separate '*-test' model, never {MODEL!r}"
        )
    return name


def champion(client: RegistryClient, name: str) -> str:
    return str(client.get_model_version_by_alias(name, "champion").version)


def time_cycle(client: RegistryClient, name: str, target: str, base: str) -> dict[str, Any]:
    """promote(target) then rollback, each wall-timed including a read-back of ``champion``."""
    t0 = time.perf_counter()
    p = promote(client, target, name)
    seen_p = champion(client, name)
    t1 = time.perf_counter()
    r = rollback(client, name)
    seen_r = champion(client, name)
    t2 = time.perf_counter()
    if seen_p != target or seen_r != base:
        raise RuntimeError(f"alias read-back mismatch: {seen_p}/{seen_r}, wanted {target}/{base}")
    return {
        "promote_s": round(p.seconds, 3),
        "promote_wall_with_readback_s": round(t1 - t0, 3),
        "rollback_s": round(r.seconds, 3),
        "rollback_wall_with_readback_s": round(t2 - t1, 3),
        "cycle_wall_s": round(t2 - t0, 3),
    }


def stats(xs: list[float]) -> dict[str, float]:
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs)}


def summarize(cycles: list[dict[str, Any]], budget_s: float = 60.0) -> dict[str, Any]:
    keys = ("promote_s", "rollback_s", "rollback_wall_with_readback_s", "cycle_wall_s")
    out: dict[str, Any] = {k: stats([c[k] for c in cycles]) for k in keys}
    worst = max(c["rollback_wall_with_readback_s"] for c in cycles)
    out["alias_move_budget_s"] = budget_s
    out["alias_move_within_budget"] = worst < budget_s
    return out


def ensure_versions(client: Any, name: str) -> tuple[str, str]:  # pragma: no cover - network
    import mlflow

    with contextlib.suppress(mlflow.exceptions.MlflowException):
        client.create_registered_model(
            name, description="Rollback timing test (PRD 10.8). Dummy versions, no weights."
        )
    existing = sorted(
        (mv for mv in client.search_model_versions(f"name='{name}'")), key=lambda m: int(m.version)
    )
    if len(existing) >= 2:
        return str(existing[0].version), str(existing[1].version)
    mlflow.set_experiment("breachlab-rollback-test")
    versions = [str(m.version) for m in existing]
    while len(versions) < 2:
        with mlflow.start_run(run_name=f"rollback-test-dummy-{len(versions) + 1}") as run:
            mlflow.log_dict({"dummy": True, "n": len(versions) + 1}, "dummy.json")
            mlflow.set_tags({"purpose": "rollback-timing-test"})
        mv = client.create_model_version(
            name, f"runs:/{run.info.run_id}/dummy.json", run.info.run_id, tags={"dummy": "true"}
        )
        versions.append(str(mv.version))
    return versions[0], versions[1]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - network
    import mlflow

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=TEST_MODEL)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="docs/lanes/P-rollback-timing.json")
    a = ap.parse_args(argv)
    name = check_name(a.model)
    client = mlflow.MlflowClient()
    t0 = time.perf_counter()
    v1, v2 = ensure_versions(client, name)
    setup_s = time.perf_counter() - t0
    client.set_registered_model_alias(name, "champion", v1)
    with contextlib.suppress(Exception):
        client.delete_registered_model_alias(name, "previous")
    cycles = [time_cycle(client, name, v2, v1) for _ in range(a.reps)]
    host = (mlflow.get_tracking_uri().split("//")[-1].split("/")[0]) or "?"
    out = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tracking_host": host,
        "model": name,
        "versions": {"base": v1, "target": v2},
        "setup_s": round(setup_s, 2),
        "reps": a.reps,
        "cycles": cycles,
        "summary": summarize(cycles),
        "note": "Alias move only (registry). Production takes effect when the gateway re-reads the alias (poll "
        "interval) and, on the Vercel guard, only if the served ONNX file changes (runbook).",
        "command": "uv run python scripts/p_rollback_timing.py",
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
