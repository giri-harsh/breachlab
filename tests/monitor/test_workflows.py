"""canary.yml and drift.yml are cheap, correct and never echo secrets; the skip gates name what is missing."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

WF = Path(__file__).resolve().parents[2] / ".github" / "workflows"
SECRET_NAMES = (
    "ADMIN_PASSWORD",
    "GATEWAY_SECRET_KEY",
    "GUARD_URL",
    "GRAFANA_OTLP_ENDPOINT",
    "GRAFANA_OTLP_INSTANCE_ID",
    "GRAFANA_OTLP_TOKEN",
    "NEON_DATABASE_URL",
)


def load(name: str) -> dict[str, Any]:
    d = yaml.safe_load((WF / f"{name}.yml").read_text())
    assert isinstance(d, dict)
    return d


def gate(name: str) -> str:
    step = next(s for s in load(name)["jobs"][name]["steps"] if s.get("id") == "gate")
    return str(step["run"])


@pytest.mark.parametrize(
    ("name", "cron", "max_timeout"), [("canary", "17 * * * *", 10), ("drift", "43 4 * * 1", 15)]
)
def test_schedule_concurrency_timeout_permissions(name: str, cron: str, max_timeout: int) -> None:
    wf = load(name)
    triggers = wf[True] if True in wf else wf["on"]  # PyYAML parses the bare key `on` as True
    assert triggers["schedule"] == [{"cron": cron}] and "workflow_dispatch" in triggers
    assert wf["concurrency"] == {"group": name, "cancel-in-progress": False}
    assert wf["permissions"] == {"contents": "read"}
    job = wf["jobs"][name]
    assert job["timeout-minutes"] <= max_timeout
    # every heavy step is gated on the secret check so a skipped run is ~10 s
    heavy = [s for s in job["steps"] if s.get("id") != "gate"]
    assert heavy and all(s.get("if") == "steps.gate.outputs.run == 'true'" for s in heavy)


@pytest.mark.parametrize("name", ["canary", "drift"])
def test_no_secret_is_echoed_or_inlined_into_shell(name: str) -> None:
    for step in load(name)["jobs"][name]["steps"]:
        run = str(step.get("run", ""))
        assert "${{ secrets" not in run and "set -x" not in run
        for line in run.splitlines():
            if re.search(r"\becho\b|\bprintf\b|\bcat\b", line):
                for v in SECRET_NAMES:
                    # names may be printed (in quotes or backticks); values ($VAR or ${VAR}) never
                    assert not re.search(rf"\$\{{?{v}\b", line), (name, line)


def run_gate(name: str, env: dict[str, str], tmp: Path) -> tuple[str, str, str]:
    out, summary = tmp / "out", tmp / "summary"
    e = {
        "PATH": os.environ["PATH"],
        "GITHUB_OUTPUT": str(out),
        "GITHUB_STEP_SUMMARY": str(summary),
        **env,
    }
    r = subprocess.run(["bash", "-e", "-c", gate(name)], env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sm = summary.read_text() if summary.exists() else ""
    return r.stdout, out.read_text(), sm


def test_canary_gate_lists_exactly_what_is_missing(tmp_path: Path) -> None:
    stdout, outputs, summary = run_gate("canary", {}, tmp_path)
    assert "run=false" in outputs
    assert "::notice title=canary skipped::" in stdout and "scripts/set_gh_secrets.sh" in stdout
    for name in ("ADMIN_PASSWORD or GATEWAY_SECRET_KEY", "GUARD_URL", "GRAFANA_OTLP_TOKEN"):
        assert name in stdout
        assert name.split(" ")[0] in summary
    assert "required (either one)" in summary and "SKIPPED" in summary


def test_canary_gate_runs_with_a_key_and_names_optional_gaps(tmp_path: Path) -> None:
    stdout, outputs, summary = run_gate(
        "canary", {"ADMIN_PASSWORD": "hunter2-VALUE", "GRAFANA_OTLP_TOKEN": "tok-VALUE"}, tmp_path
    )
    assert "run=true" in outputs and not summary
    assert "unset (optional): GUARD_URL, GRAFANA_OTLP_ENDPOINT, GRAFANA_OTLP_INSTANCE_ID" in stdout
    assert "GRAFANA_OTLP_TOKEN," not in stdout and "VALUE" not in stdout  # values never printed


def test_canary_gate_silent_when_everything_is_set(tmp_path: Path) -> None:
    env = {v: "x-VALUE" for v in SECRET_NAMES}
    stdout, outputs, summary = run_gate("canary", env, tmp_path)
    assert "run=true" in outputs and stdout.strip() == "" and not summary


def test_drift_gate_names_the_missing_secret_and_never_prints_values(tmp_path: Path) -> None:
    stdout, outputs, summary = run_gate("drift", {}, tmp_path)
    assert (
        "run=false" in outputs and "NEON_DATABASE_URL" in stdout and "NEON_DATABASE_URL" in summary
    )
    assert "scripts/set_gh_secrets.sh" in summary
    d2 = tmp_path / "second"
    d2.mkdir()
    stdout, outputs, _ = run_gate("drift", {"NEON_DATABASE_URL": "postgres://u:p-VALUE@h/db"}, d2)
    assert "run=true" in outputs and "VALUE" not in stdout
    assert "GRAFANA_OTLP_* not set" in stdout


def test_canary_workflow_passes_the_new_guard_knob_and_runs_the_module() -> None:
    job = load("canary")["jobs"]["canary"]
    assert job["env"]["CANARY_REQUIRE_GUARD"] == "${{ vars.CANARY_REQUIRE_GUARD }}"
    last = job["steps"][-1]
    assert last["run"] == "uv run --no-sync python -m breachlab.monitor.canary"
