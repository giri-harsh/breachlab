"""Check that the custom domain is live: DNS, HTTPS, API CORS and (optionally) Vercel's own status.

    python scripts/verify_domain.py [--domain breachlab.harshgiri.site] [--api https://breachlab-api.vercel.app]
        [--wait 600] [--list-links] [--write-links docs/lanes/C-domain-links.txt]

Exit code 0 only if every check prints PASS. Checks:
  1. DNS    the domain is a CNAME to the Vercel target (or resolves to a Vercel IP). Uses dnspython when installed, else
            DNS-over-HTTPS (dns.google, works where only HTTPS egress is allowed), else ``nslookup``, else the OS resolver.
  2. HTTPS  GET https://<domain>/ answers 200 with a valid certificate; GET https://<domain>/api/health is reported
            (the web project does not serve /api/health, the gateway does: 404 here is expected, so it is INFO only).
  3. CORS   a browser preflight from https://<domain> to the gateway's /api/message is allowed
            (needs CORS_ORIGIN on breachlab-api to include the new origin).
  4. VERCEL (only with VERCEL_TOKEN set) the project domain is verified and not misconfigured.
``--list-links`` prints the files that still contain the old site URL (see docs/lanes/C-domain.md); it changes nothing.
The token is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "breachlab.harshgiri.site"
API = "https://breachlab-api.vercel.app"
OLD_URL = "https://breachlab-web.vercel.app"
TEAM = "team_yIXRPPOQWbqXkM682GSKqORh"
PROJECT = "prj_mWcWKnDvOQfVL99vf9MHbCIgWR8P"
# From GET /v6/domains/<domain>/config (recommendedCNAME, recommendedIPv4), 2026-11-13.
CNAME_TARGETS = {"d451dc7baa80fa91.vercel-dns-017.com", "cname.vercel-dns.com"}
VERCEL_IPS = {"216.198.79.1", "64.29.17.1", "76.76.21.21"}


def _get(
    url: str, headers: dict[str, str] | None = None, method: str = "GET", timeout: float = 15.0
) -> tuple[int, dict[str, str], str]:
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - https only, fixed hosts
            return (
                r.status,
                {k.lower(): v for k, v in r.headers.items()},
                r.read(200_000).decode("utf-8", "replace"),
            )
    except urllib.error.HTTPError as e:
        return (
            e.code,
            {k.lower(): v for k, v in e.headers.items()},
            e.read(200_000).decode("utf-8", "replace"),
        )


def dns_records(name: str, rtype: str) -> tuple[list[str], str]:
    """Answers (lower-case, no trailing dot) and the method used."""
    try:
        import dns.resolver  # type: ignore[import-not-found,unused-ignore]

        try:
            ans = dns.resolver.resolve(name, rtype)
            return [
                str(r.target if rtype == "CNAME" else r).rstrip(".").lower() for r in ans
            ], "dnspython"
        except Exception:  # noqa: BLE001 - NXDOMAIN, NoAnswer, timeouts: no record
            return [], "dnspython"
    except ImportError:
        pass
    try:
        q = urllib.parse.urlencode({"name": name, "type": rtype})
        _, _, body = _get(f"https://dns.google/resolve?{q}")
        code = {"A": 1, "CNAME": 5}[rtype]
        data = json.loads(body)
        return [
            a["data"].rstrip(".").lower() for a in data.get("Answer", []) if a.get("type") == code
        ], "doh:dns.google"
    except (OSError, ValueError):
        pass
    if shutil.which("nslookup"):
        out = subprocess.run(
            ["nslookup", f"-type={rtype}", name],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout  # noqa: S603
        pat = r"canonical name = (\S+?)\.?$" if rtype == "CNAME" else r"^Address:\s*([0-9.]+)\s*$"
        found = re.findall(pat, out, re.M)
        return [
            x.lower() for x in found if rtype == "CNAME" or not x.startswith("127.")
        ], "nslookup"
    if rtype == "A":
        try:
            return sorted(
                {i[4][0] for i in socket.getaddrinfo(name, 443, socket.AF_INET)}
            ), "socket"
        except OSError:
            return [], "socket"
    return [], "none"


def check_dns(domain: str) -> tuple[bool, str]:
    cn, how = dns_records(domain, "CNAME")
    if cn:
        ok = cn[0] in CNAME_TARGETS
        return ok, f"CNAME {domain} -> {cn[0]} ({how}); expected one of {sorted(CNAME_TARGETS)}"
    a, how = dns_records(domain, "A")
    if a:
        ok = bool(set(a) & VERCEL_IPS)
        return ok, f"A {domain} -> {a} ({how}); Vercel IPs {sorted(VERCEL_IPS)}"
    return (
        False,
        f"no CNAME or A record for {domain} yet ({how}); add the CNAME from docs/blockers.md",
    )


def check_https(domain: str) -> tuple[bool, str]:
    try:
        status, _, body = _get(f"https://{domain}/")
    except (OSError, ValueError) as e:
        return False, f"https://{domain}/ unreachable: {type(e).__name__}: {e}"
    ok = status == 200 and "BreachLab" in body
    return (
        ok,
        f"GET https://{domain}/ -> {status}{'' if 'BreachLab' in body else ' (page does not mention BreachLab)'}",
    )


def info_health(domain: str) -> str:
    try:
        status, _, body = _get(f"https://{domain}/api/health")
        return (
            f"INFO GET https://{domain}/api/health -> {status} (the web project does not serve it; the gateway does) {body[:80]!r}"
            if status != 200
            else f"INFO GET https://{domain}/api/health -> 200 {body[:80]!r}"
        )
    except (OSError, ValueError) as e:
        return f"INFO https://{domain}/api/health unreachable: {type(e).__name__}"


def check_cors(domain: str, api: str) -> tuple[bool, str]:
    origin = f"https://{domain}"
    try:
        status, hdr, body = _get(
            f"{api.rstrip('/')}/api/message",
            {
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
            method="OPTIONS",
        )
    except (OSError, ValueError) as e:
        return False, f"preflight to {api} failed: {type(e).__name__}: {e}"
    allow = hdr.get("access-control-allow-origin", "")
    ok = status == 200 and allow in (origin, "*")
    hint = (
        ""
        if ok
        else f" ({body.strip()[:40]}); fix: add the origin to CORS_ORIGIN (comma separated) on breachlab-api and redeploy"
    )
    return (
        ok,
        f"OPTIONS {api}/api/message Origin {origin} -> {status}, allow-origin={allow!r}{hint}",
    )


def check_vercel(domain: str) -> tuple[bool, str] | None:
    token = os.environ.get("VERCEL_TOKEN")
    if not token:
        return None
    h = {"Authorization": f"Bearer {token}"}
    try:
        _, _, b1 = _get(
            f"https://api.vercel.com/v9/projects/{PROJECT}/domains/{domain}?teamId={TEAM}", h
        )
        _, _, b2 = _get(f"https://api.vercel.com/v6/domains/{domain}/config?teamId={TEAM}", h)
        d1, d2 = json.loads(b1), json.loads(b2)
    except (OSError, ValueError) as e:
        return False, f"Vercel API unreachable: {type(e).__name__}"
    ok = bool(d1.get("verified")) and d2.get("misconfigured") is False
    return ok, f"Vercel: verified={d1.get('verified')}, misconfigured={d2.get('misconfigured')}"


def old_url_files() -> list[tuple[str, int]]:
    try:
        out = subprocess.run(
            ["git", "grep", "-c", OLD_URL], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout  # noqa: S603, S607
        rows = [(p, int(n)) for p, _, n in (line.rpartition(":") for line in out.splitlines())]
    except OSError:
        rows = []
    return sorted(rows)


def render_links(rows: list[tuple[str, int]]) -> str:
    history = ("docs/decisions.md", "docs/progress.md", "docs/lanes/")
    code = [r for r in rows if r[0].startswith(("web/", "scripts/", "src/"))]
    kept = [r for r in rows if r[0].startswith(history)]
    docs = [r for r in rows if r not in code and r not in kept]
    lines = [
        f"Files that contain the old site URL {OLD_URL} (git grep -c). Do NOT change them until {DOMAIN} is live",
        "(python scripts/verify_domain.py prints PASS for every check). Generated by scripts/verify_domain.py --write-links.",
        "",
        "UPDATE when live (code and scripts that build links or set production env):",
    ]
    lines += [f"  {p}  ({n} occurrence{'s' if n != 1 else ''})" for p, n in code]
    lines += [
        "",
        "UPDATE when live (README, release notes, runbook, contracts, resume docs and tests):",
    ]
    lines += [f"  {p}  ({n} occurrence{'s' if n != 1 else ''})" for p, n in docs]
    lines += [
        "",
        "KEEP as history (entries describe what was true on that date; add a new line instead of editing):",
    ]
    lines += [f"  {p}  ({n} occurrence{'s' if n != 1 else ''})" for p, n in kept]
    lines += [
        "",
        "Keep the vercel.app URL working as an alias: do not remove it from the Vercel project, and keep it in CORS_ORIGIN as long as old links circulate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=DOMAIN)
    ap.add_argument("--api", default=API)
    ap.add_argument(
        "--wait", type=int, default=0, help="poll DNS for up to this many seconds before checking"
    )
    ap.add_argument("--list-links", action="store_true")
    ap.add_argument("--write-links", metavar="PATH")
    a = ap.parse_args()
    if a.list_links or a.write_links:
        text = render_links(old_url_files())
        if a.write_links:
            Path(a.write_links).write_text(text)
        else:
            print(text)
        if a.write_links or a.list_links:
            return 0
    deadline = time.time() + a.wait
    while True:
        dns_ok, dns_msg = check_dns(a.domain)
        if dns_ok or time.time() >= deadline:
            break
        print(f"waiting for DNS: {dns_msg}", flush=True)
        time.sleep(30)
    results = [("DNS", dns_ok, dns_msg)]
    ok, msg = check_https(a.domain)
    results.append(("HTTPS", ok, msg))
    ok, msg = check_cors(a.domain, a.api)
    results.append(("CORS", ok, msg))
    v = check_vercel(a.domain)
    if v is not None:
        results.append(("VERCEL", v[0], v[1]))
    for name, passed, text in results:
        print(f"{'PASS' if passed else 'FAIL'} {name}: {text}")
    print(info_health(a.domain))
    all_ok = all(r[1] for r in results)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
