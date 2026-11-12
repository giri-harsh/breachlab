"""Create/refresh the `breachlab-guard` Vercel project and deploy the guard function (ADR-0080, lane B3).

    VERCEL_TOKEN=... HF_TOKEN=... [GUARD_TOKEN=...] python scripts/deploy_vercel_guard.py [--no-deploy] [--onnx-file guard-q4b32.onnx]

Steps: (1) the project exists (REST create if not) with Deployment Protection off, because the gateway calls it
server to server and `/score` is guarded by the bearer token instead; (2) runtime env is upserted over REST
(`HF_TOKEN`, `GUARD_TOKEN` as write-only `sensitive`, the rest `encrypted`); values come from this process'
environment and are never printed; (3) the project is staged into a folder without version-control metadata
(ADR-0055) and deployed with the Vercel CLI to production. Prints the production URL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

API = "https://api.vercel.com"
TEAM = "team_yIXRPPOQWbqXkM682GSKqORh"
SCOPE = "harshs-projects-d6e22d93"
NAME = "breachlab-guard"  # default project; ``--name breachlab-guard-base`` deploys the L3 host
SECRETS = ("HF_TOKEN", "GUARD_TOKEN")
PLAIN = (
    "GUARD_MODEL_REPO",
    "GUARD_ONNX_FILE",
    "GUARD_CALIBRATION_FILE",
    "GUARD_THREADS",
    "GUARD_ALIAS",
)


def call(client: httpx.Client, method: str, path: str, **kw: Any) -> httpx.Response:
    params = {"teamId": TEAM, **kw.pop("params", {})}
    return client.request(method, f"{API}{path}", params=params, **kw)


def ensure_project(client: httpx.Client, name: str = NAME) -> str:
    r = call(client, "GET", f"/v9/projects/{name}")
    if r.status_code == 404:
        r = call(client, "POST", "/v11/projects", json={"name": name, "framework": None})
    r.raise_for_status()
    pid = str(r.json()["id"])
    call(client, "PATCH", f"/v9/projects/{pid}", json={"ssoProtection": None}).raise_for_status()
    return pid


def upsert_env(client: httpx.Client, pid: str, env: dict[str, str]) -> list[str]:
    done: list[str] = []
    for key in (*SECRETS, *PLAIN):
        value = env.get(key)
        if not value:
            continue
        body = {
            "key": key,
            "value": value,
            "type": "sensitive" if key in SECRETS else "encrypted",
            "target": ["production"],
        }
        r = call(client, "POST", f"/v10/projects/{pid}/env", params={"upsert": "true"}, json=body)
        r.raise_for_status()
        done.append(key)
    return done


def deploy(pid: str, name: str = NAME) -> str:
    from breachlab.serve.vercel_build import stage

    with tempfile.TemporaryDirectory(prefix="breachlab-guard-") as tmp:
        dest = Path(tmp) / "project"
        stage(dest)
        (dest / ".vercel").mkdir()
        (dest / ".vercel" / "project.json").write_text(
            json.dumps({"projectId": pid, "orgId": TEAM}), encoding="utf-8"
        )
        out = subprocess.run(
            ["npx", "--yes", "vercel@latest", "deploy", "--prod", "--yes", "--scope", SCOPE],
            cwd=dest,
            capture_output=True,
            text=True,
            check=False,
        )
    if out.returncode:
        raise SystemExit("vercel deploy failed:\n" + (out.stdout + out.stderr)[-1500:])
    for line in out.stdout.splitlines():
        if '"productionUrl"' in line:
            return line.split('"')[3]
    return f"https://{name}.vercel.app"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-deploy", action="store_true", help="project + env only")
    ap.add_argument("--onnx-file", default=None, help="file under <alias>/ in the model repo")
    ap.add_argument(
        "--name", default=NAME, help="Vercel project (breachlab-guard-base for the L3 host)"
    )
    ap.add_argument(
        "--alias",
        default=None,
        help="Hub folder and served alias (champion, or base for the L3 host)",
    )
    ap.add_argument(
        "--calibration-file",
        default=None,
        help="calibration JSON under <alias>/ (e.g. calibration-q4b32.json)",
    )
    a = ap.parse_args(argv)
    env = dict(os.environ)
    if a.onnx_file:
        env["GUARD_ONNX_FILE"] = a.onnx_file
    if a.alias:
        env["GUARD_ALIAS"] = a.alias
    if a.calibration_file:
        env["GUARD_CALIBRATION_FILE"] = a.calibration_file
    token = env.get("VERCEL_TOKEN")
    if not token:
        raise SystemExit("VERCEL_TOKEN is not set")
    with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=60.0) as client:
        pid = ensure_project(client, a.name)
        print("project", a.name, pid)
        print("env set:", ", ".join(upsert_env(client, pid, env)) or "(none)")
    if not a.no_deploy:
        print("production:", deploy(pid, a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
