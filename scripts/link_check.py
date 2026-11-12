"""Check that every public link in README.md and the web app responds 200.

Web links are read from `web/**/*.tsx|ts` string literals plus the constants in `web/lib/links.ts`
(`NAME = process.env.X ?? "default"` defaults are substituted into the template URLs; URLs that stay unresolved,
e.g. an unset optional Space, are skipped). While the repo/HF artifacts are private a 401/403/404 on those hosts is a
WARN, not a failure, unless LINKCHECK_STRICT=1 (set it after the public release, Phase 7).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
URL = re.compile(r"https?://[^\s)>\"'`\\]+")
SKIP = ("localhost", "127.0.0.1", "example.com", "placeholder.invalid")
DEFAULT = re.compile(r"export const (\w+)\s*=\s*process\.env\.\w+\s*\?\?\s*\"([^\"]*)\"")
PRIVATE_HOSTS = ("github.com/", "huggingface.co/")


def _clean(urls: list[str]) -> list[str]:
    return [u.rstrip(".,;:") for u in urls if not any(s in u for s in SKIP)]


def readme_links(root: Path = ROOT) -> list[str]:
    f = root / "README.md"
    return _clean(URL.findall(f.read_text(encoding="utf-8"))) if f.exists() else []


def web_defaults(root: Path = ROOT) -> dict[str, str]:
    f = root / "web" / "lib" / "links.ts"
    return dict(DEFAULT.findall(f.read_text(encoding="utf-8"))) if f.exists() else {}


def resolve(url: str, defaults: dict[str, str]) -> str | None:
    """Substitute ``${NAME}`` placeholders from the defaults; None when any stays unknown or empty."""
    missing = False

    def sub(m: re.Match[str]) -> str:
        nonlocal missing
        val = defaults.get(m.group(1))
        if not val:
            missing = True
            return ""
        return val

    out = re.sub(r"\$\{(\w+)\}", sub, url)
    return None if missing else out


def web_links(root: Path = ROOT) -> list[str]:
    defaults = web_defaults(root)
    found: list[str] = []
    web = root / "web"
    for pattern in ("app/**/*.tsx", "components/**/*.tsx", "lib/**/*.ts"):
        for f in sorted(web.glob(pattern)):
            for raw in URL.findall(f.read_text(encoding="utf-8")):
                if "${" in raw and (r := resolve(raw, defaults)) is not None:
                    found.append(r)
                elif "${" not in raw:
                    found.append(raw)
    return _clean(found)


def links(root: Path = ROOT) -> list[str]:
    return sorted(set(readme_links(root)) | set(web_links(root)))


def probe_url(url: str) -> str:
    """The gateway has no page at its root: probe its health route instead."""
    if re.fullmatch(r"https://[\w-]+-api\.vercel\.app/?", url):
        return url.rstrip("/") + "/api/health"
    return url


def classify(url: str, status: int | None, strict: bool) -> str:
    """OK / WARN (private host not reachable anonymously) / FAIL."""
    if status == 200:
        return "OK"
    if not strict and status in (401, 403, 404) and any(h in url for h in PRIVATE_HOSTS):
        return "WARN"
    return "FAIL"


def main() -> int:
    strict = os.environ.get("LINKCHECK_STRICT") == "1"
    bad = 0
    for u in links():
        try:
            status: int | None = httpx.get(
                probe_url(u), follow_redirects=True, timeout=20
            ).status_code
            detail = str(status)
        except httpx.HTTPError as exc:
            status, detail = None, type(exc).__name__
        verdict = classify(u, status, strict)
        print(f"{verdict:4}", u, "" if verdict == "OK" else detail)
        bad += verdict == "FAIL"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
