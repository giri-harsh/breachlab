"""One-command hand-off once a guard host exists (ADR-0063).

    GUARD_URL=https://<host> VERCEL_TOKEN=... [GUARD_TOKEN=...] python scripts/set_guard_url.py [--ref main] [--no-live] [--dry-run]

Steps (each stops the run on failure, nothing is changed before the host validates):
  1. validate the URL shape (https, a real host name: not the `.invalid` placeholder, localhost or a private IP literal);
  2. GET {url}/health must answer 200 with {"ok": true, "models": {"champion": <version>}} and POST {url}/score must return a
     well-formed verdict (label, category, p_unsafe in [0, 1]); cold starts are retried with backoff;
  3. set GUARD_URL (and GUARD_TOKEN when given, stored write-only) on the `breachlab-api` Vercel project through the
     REST API (values are never printed); the web app learns the guard status at runtime from /api/health (no build flag);
  4. redeploy both projects from git (`--ref`, default main) and wait until they are READY;
  5. run the live Playwright spec (levels 1-5) against the web deployment unless `--no-live`.
`--dry-run` stops after step 2. Secrets only come from the environment.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
VERCEL = "https://api.vercel.com"
TEAM = "team_yIXRPPOQWbqXkM682GSKqORh"
API_PROJECT = ("breachlab-api", "prj_xSY0LQ5tzcH3fdGT7EbDXbvk551N")
WEB_PROJECT = ("breachlab-web", "prj_mWcWKnDvOQfVL99vf9MHbCIgWR8P")
REPO_ID = 1438652749
WEB_URL = "https://breachlab-web.vercel.app"
CATEGORIES = {"none", "injection", "jailbreak", "exfiltration"}


class HandoffError(RuntimeError):
    """A validation or deployment step failed; the message is safe to print (no secrets)."""


def validate_url(raw: str) -> str:
    """Return the normalised base URL (no trailing slash) or raise HandoffError."""
    url = (raw or "").strip().rstrip("/")
    if not url:
        raise HandoffError("GUARD_URL is empty")
    p = urlparse(url)
    if p.scheme != "https":
        raise HandoffError("GUARD_URL must start with https://")
    host = (p.hostname or "").lower()
    if not host or "." not in host:
        raise HandoffError("GUARD_URL has no usable host name")
    if host == "localhost" or host.endswith(
        (".invalid", ".local", ".localhost", ".test", ".example")
    ):
        raise HandoffError(f"GUARD_URL host {host!r} is a placeholder or local name")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        raise HandoffError("GUARD_URL is a private or loopback address")
    if p.path not in ("", "/") or p.query or p.fragment or p.username or p.password:
        raise HandoffError(
            "GUARD_URL must be the bare service root (no path, query or credentials)"
        )
    return url


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def check_health(
    client: httpx.Client,
    url: str,
    token: str | None = None,
    attempts: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """GET /health until it answers 200 with the contract body (a cold host may need a few tries)."""
    last = "no attempt"
    for i in range(attempts):
        try:
            r = client.get(f"{url}/health", headers=_headers(token))
            if r.status_code == 200:
                body = r.json()
                models = body.get("models") if isinstance(body, dict) else None
                if not (
                    isinstance(body, dict)
                    and body.get("ok") is True
                    and isinstance(models, dict)
                    and models.get("champion")
                ):
                    raise HandoffError(
                        '/health answered 200 but not {"ok": true, "models": {"champion": ...}}'
                    )
                return dict(body)
            last = f"HTTP {r.status_code}"
            if r.status_code in (401, 403):
                raise HandoffError(
                    f"/health refused the request (HTTP {r.status_code}): set GUARD_TOKEN for a private host"
                )
        except httpx.HTTPError as e:
            last = type(e).__name__
        except ValueError:
            raise HandoffError("/health did not return JSON") from None
        if i < attempts - 1:
            sleep(2.0 * (2**i))
    raise HandoffError(f"/health never became healthy ({last})")


def check_score(client: httpx.Client, url: str, token: str | None = None) -> dict[str, Any]:
    """POST /score with a benign text and validate the verdict against docs/CONTRACTS.md section 1."""
    r = client.post(
        f"{url}/score",
        json={"text": "Hello, what can you do?", "kind": "input", "alias": "champion"},
        headers=_headers(token),
    )
    if r.status_code != 200:
        raise HandoffError(f"/score answered HTTP {r.status_code}")
    try:
        d = r.json()
    except ValueError:
        raise HandoffError("/score did not return JSON") from None
    p = d.get("p_unsafe") if isinstance(d, dict) else None
    if not (
        isinstance(d, dict)
        and d.get("label") in ("SAFE", "UNSAFE")
        and d.get("category") in CATEGORIES
        and isinstance(p, int | float)
        and 0.0 <= p <= 1.0
    ):
        raise HandoffError("/score body does not match the guard contract")
    return dict(d)


class Vercel:
    def __init__(self, token: str, client: httpx.Client | None = None) -> None:
        self.c = client or httpx.Client(timeout=60)
        self.h = {"Authorization": f"Bearer {token}"}

    def upsert_env(
        self, project_id: str, key: str, value: str, *, sensitive: bool, targets: list[str]
    ) -> None:
        r = self.c.post(
            f"{VERCEL}/v10/projects/{project_id}/env",
            params={"teamId": TEAM, "upsert": "true"},
            headers=self.h,
            json={
                "key": key,
                "value": value,
                "type": "sensitive" if sensitive else "encrypted",
                "target": targets,
            },
        )
        if r.status_code not in (200, 201):
            raise HandoffError(f"setting {key} failed: HTTP {r.status_code}")

    def redeploy(self, name: str, project_id: str, ref: str) -> str:
        r = self.c.post(
            f"{VERCEL}/v13/deployments",
            params={"teamId": TEAM, "forceNew": "1", "skipAutoDetectionConfirmation": "1"},
            headers=self.h,
            json={
                "name": name,
                "project": project_id,
                "target": "production",
                "gitSource": {"type": "github", "repoId": REPO_ID, "ref": ref},
            },
        )
        if r.status_code != 200:
            raise HandoffError(f"redeploy of {name} failed: HTTP {r.status_code}")
        return str(r.json()["id"])

    def wait_ready(
        self, dpl: str, timeout: float = 600, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            r = self.c.get(
                f"{VERCEL}/v13/deployments/{dpl}", params={"teamId": TEAM}, headers=self.h
            )
            st = r.json().get("readyState") if r.status_code == 200 else None
            if st == "READY":
                return
            if st in ("ERROR", "CANCELED"):
                raise HandoffError(f"deployment {dpl} ended in state {st}")
            sleep(8)
        raise HandoffError(f"deployment {dpl} not READY after {int(timeout)} s")


def run_live_spec(web_url: str = WEB_URL) -> int:
    """Levels 1-5 through the real UI; E2E_EXPECT_GUARD_OFFLINE is deliberately removed so an offline guard FAILS."""
    env = {k: v for k, v in os.environ.items() if k != "E2E_EXPECT_GUARD_OFFLINE"}
    env.update({"E2E_BASE_URL": web_url, "E2E_LIVE": "1"})
    env.setdefault("PW_CHROMIUM", "/opt/pw-browsers/chromium")
    return subprocess.run(
        ["npx", "playwright", "test", "live"], cwd=ROOT / "web", env=env, check=False
    ).returncode


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", default="main", help="git ref to redeploy (default main)")
    ap.add_argument("--no-live", action="store_true", help="skip the live Playwright spec")
    ap.add_argument("--dry-run", action="store_true", help="validate the host only; change nothing")
    a = ap.parse_args(argv)
    env = dict(os.environ if environ is None else environ)
    try:
        url = validate_url(env.get("GUARD_URL", ""))
        guard_token = env.get("GUARD_TOKEN") or None
        with httpx.Client(timeout=30, follow_redirects=False) as c:
            health = check_health(c, url, guard_token)
            print(f"health ok: champion={health['models'].get('champion')}")
            verdict = check_score(c, url, guard_token)
            print(f"score ok: label={verdict['label']} p_unsafe={verdict['p_unsafe']}")
        if a.dry_run:
            print("dry run: nothing changed")
            return 0
        vtoken = env.get("VERCEL_TOKEN")
        if not vtoken:
            raise HandoffError("VERCEL_TOKEN is not set")
        v = Vercel(vtoken)
        v.upsert_env(API_PROJECT[1], "GUARD_URL", url, sensitive=False, targets=["production"])
        if guard_token:
            v.upsert_env(
                API_PROJECT[1], "GUARD_TOKEN", guard_token, sensitive=True, targets=["production"]
            )
        print("env set (GUARD_URL)")
        ids = [v.redeploy(n, p, a.ref) for n, p in (API_PROJECT, WEB_PROJECT)]
        for dpl in ids:
            v.wait_ready(dpl)
        print(f"redeployed from {a.ref}: {', '.join(ids)}")
    except HandoffError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2
    if a.no_live:
        return 0
    rc = run_live_spec()
    print("live spec", "passed" if rc == 0 else f"FAILED (exit {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
