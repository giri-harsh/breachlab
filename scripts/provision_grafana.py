"""Provision dashboards and the D6 alert rules (no contact point) into Grafana Cloud.

Needs GRAFANA_URL (https://<stack>.grafana.net) and GRAFANA_API_TOKEN (service-account token, Editor+).
The OTLP credentials in the env cannot call the Grafana HTTP API, so without those two this exits with a
clear message (docs/blockers.md). Prometheus datasource UID: GRAFANA_PROM_UID or the first prometheus source.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1] / "dashboards"


def main() -> int:
    url, tok = os.environ.get("GRAFANA_URL"), os.environ.get("GRAFANA_API_TOKEN")
    if not url or not tok:
        print(
            "GRAFANA_URL and GRAFANA_API_TOKEN are required (see docs/blockers.md)", file=sys.stderr
        )
        return 2
    h = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    c = httpx.Client(base_url=url.rstrip("/"), headers=h, timeout=30)
    uid = os.environ.get("GRAFANA_PROM_UID")
    if not uid:
        ds = c.get("/api/datasources").raise_for_status().json()
        uid = next(d["uid"] for d in ds if d["type"] == "prometheus")
    folder = c.post("/api/folders", json={"title": "BreachLab"})
    folder_uid = (
        folder.json().get("uid")
        if folder.status_code < 300
        else next(f["uid"] for f in c.get("/api/folders").json() if f["title"] == "BreachLab")
    )
    for f in sorted(ROOT.glob("*.json")):
        if f.name == "alerts.json":
            continue
        body = json.loads(f.read_text().replace("${DS}", uid))
        body["templating"] = {"list": []}
        c.post(
            "/api/dashboards/db",
            json={"dashboard": body, "folderUid": folder_uid, "overwrite": True},
        ).raise_for_status()
        print("dashboard", f.stem)
    spec = json.loads(ROOT.joinpath("alerts.json").read_text().replace("${DS_UID}", uid))
    group = {"title": spec["group"], "interval": 60, "folderUID": folder_uid, "rules": []}
    for r in spec["rules"]:
        r = dict(r)
        r.update(
            {"folderUID": folder_uid, "ruleGroup": spec["group"], "orgID": 1, "isPaused": False}
        )
        group["rules"].append(r)
    c.put(
        f"/api/v1/provisioning/folder/{folder_uid}/rule-groups/{spec['group']}",
        json=group,
        headers={**h, "X-Disable-Provenance": "true"},
    ).raise_for_status()
    print("alert rules", [r["title"] for r in spec["rules"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
