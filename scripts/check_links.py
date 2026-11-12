"""Check the links in README.md and docs/*.md (stdlib only).

    python scripts/check_links.py [--offline] [--root DIR] [--timeout 10]

* Relative links (``[text](path#anchor)``) are checked on disk with no network: the target must exist, and for a
  Markdown target the ``#anchor`` must match a heading (GitHub slug rules) in it.
* https/http links (inline links, ``<autolinks>`` and bare URLs, fenced code blocks excluded) get a HEAD request with a
  timeout of 10 s and one retry on a transport error or a 5xx. A 405 means the server is there but refuses HEAD and
  counts as reachable. Anything else >= 400 is a failure and is printed with its URL and status.
* Private repositories are never requested: ``github.com/giri-harsh/Breach-lab`` and Hugging Face ``Pseudo-Harsh``
  repos/Spaces answer 401/404 anonymously, so they are reported as "private, not checked" instead of failing.
* A bare base URL of our two API hosts (no route at "/") is probed through ``/api/health`` or ``/health``; Kaggle kernels
  of ``giriharsh`` are private too (``is_private=True`` in ``kaggle_job.py``).
* Placeholders (``localhost``, ``example.*``, ``*.invalid``, ``<host>``, ``{VAR}``, ``...``) are skipped.

Exit code 0 when no relative link is broken and no external link failed, 1 otherwise. ``--offline`` skips every
external request (they are listed as "not checked (offline)").
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

FENCE = re.compile(r"^\s*(```|~~~)")
INLINE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
BARE = re.compile(r"https?://[^\s)>\]\"'`\\|]+")
HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
HTML_ID = re.compile(r"""(?:id|name)=["']([^"']+)["']""")
SKIP_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")
PLACEHOLDER_MARKS = ("<", ">", "{", "}", "$", "...", "…")
PRIVATE = (
    re.compile(r"^https?://github\.com/giri-harsh/Breach-lab(?:[/#?].*)?$", re.I),
    re.compile(
        r"^https?://(?:www\.)?huggingface\.co/(?:(?:datasets|spaces|models)/)?Pseudo-Harsh(?:/.*)?$",
        re.I,
    ),
    re.compile(r"^https?://pseudo-harsh-[^./]+\.hf\.space(?:/.*)?$", re.I),
    # Kaggle kernels are pushed with is_private=True (src/breachlab/train/kaggle_job.py): anonymous GET answers 404.
    re.compile(r"^https?://(?:www\.)?kaggle\.com/code/giriharsh/.*$", re.I),
)
# Documentation-range example addresses (RFC 5737), e.g. 203-0-113-7.sslip.io in docs/deploy-guard.md.
EXAMPLE_HOST = re.compile(r"^(?:203-0-113|198-51-100|192-0-2)-\d+\.sslip\.io$")
# Our two API hosts have no route at "/" (GET / answers 404 by design, checked 2026-11-01), so a bare base URL is
# checked through its health route instead.
HEALTH_ROUTE = {
    "breachlab-api.vercel.app": "/api/health",
    "breachlab-guard.vercel.app": "/health",
}

# (url, timeout) -> HTTP status; raises OSError on a transport failure.
Head = Callable[[str, float], int]


@dataclass
class Report:
    broken: list[str] = field(default_factory=list)  # relative link problems
    failed: list[tuple[str, str, list[str]]] = field(default_factory=list)  # url, status, files
    private: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    ok_urls: int = 0
    relative_ok: int = 0
    unchecked: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.broken and not self.failed


def slugify(heading: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation, spaces to hyphens (inline markup removed first)."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("*", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    found: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found.update(HTML_ID.findall(line))
        m = HEADING.match(line)
        if m:
            slug = slugify(m.group(2))
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            found.add(slug if n == 0 else f"{slug}-{n}")
    return found


def prose_lines(text: str) -> list[str]:
    """Lines outside fenced code blocks."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return out


def extract(text: str) -> tuple[list[str], list[str]]:
    """(link targets from ``[t](x)`` incl. relative ones, bare/auto http(s) URLs), fenced code excluded."""
    targets: list[str] = []
    urls: list[str] = []
    for line in prose_lines(text):
        targets.extend(INLINE.findall(line))
        urls.extend(AUTOLINK.findall(line))
        stripped = INLINE.sub(lambda m: " " + m.group(1) + " ", line)
        urls.extend(u.rstrip(".,;:!?") for u in BARE.findall(stripped))
    return targets, urls


def is_external(target: str) -> bool:
    return target.lower().startswith(("http://", "https://"))


def classify(url: str) -> str:
    """'private', 'skip' or 'check'."""
    if any(p.match(url) for p in PRIVATE):
        return "private"
    host = (urlsplit(url).hostname or "").lower()
    if (
        not host
        or host in SKIP_HOSTS
        or host == "invalid"
        or host.endswith(".invalid")
        or host == "example.com"
        or host.startswith("example.")
        or EXAMPLE_HOST.match(host) is not None
        or any(m in url for m in PLACEHOLDER_MARKS)
    ):
        return "skip"
    return "check"


def probe_url(url: str) -> str:
    """The URL actually requested: a bare API base URL is probed through its health route."""
    parts = urlsplit(url)
    route = HEALTH_ROUTE.get((parts.hostname or "").lower())
    if route and parts.path in ("", "/") and not parts.query:
        return f"{parts.scheme}://{parts.netloc}{route}"
    return url


def check_relative(md: Path, target: str, root: Path) -> str | None:
    """None when fine, else the problem text. Pure filesystem."""
    if target.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return None
    parts = urlsplit(target)
    path_part, frag = unquote(parts.path), unquote(parts.fragment)
    if not path_part:  # "#anchor" inside the same file
        dest = md
    else:
        dest = (
            (root / path_part.lstrip("/")) if path_part.startswith("/") else (md.parent / path_part)
        )
        try:
            dest = dest.resolve()
        except OSError:
            return f"cannot resolve {target}"
        if not dest.exists():
            return f"missing file {path_part}"
    if (
        frag
        and dest.is_file()
        and dest.suffix.lower() == ".md"
        and frag.lower() not in {a.lower() for a in anchors(dest)}
    ):
        return f"missing anchor #{frag} in {dest.name}"
    return None


def http_head(url: str, timeout: float) -> int:
    req = urllib.request.Request(  # noqa: S310 (scheme is checked by classify/is_external)
        url, method="HEAD", headers={"User-Agent": "breachlab-check-links/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def check_external(url: str, head: Head, timeout: float = 10.0) -> tuple[bool, str]:
    """HEAD with one retry on a transport error or 5xx. Returns (ok, status text)."""
    status = "?"
    for attempt in (1, 2):
        try:
            code = head(url, timeout)
        except (OSError, ValueError) as e:  # URLError, timeout, ssl, connection reset
            status = f"error {type(e).__name__}: {e}"[:200]
            continue
        status = str(code)
        if code < 400 or code == 405:
            return True, status if code != 405 else "405 (HEAD not allowed, host reachable)"
        if code < 500:
            return False, status
        if attempt == 2:
            return False, status
    return False, status


def doc_files(root: Path) -> list[Path]:
    files = [root / "README.md"] if (root / "README.md").exists() else []
    docs = root / "docs"
    if docs.is_dir():
        files.extend(sorted(docs.glob("*.md")))
    return files


def run(
    root: Path = ROOT, head: Head | None = None, offline: bool = False, timeout: float = 10.0
) -> Report:
    rep = Report()
    head = head or http_head
    ext: dict[str, list[str]] = {}
    for md in doc_files(root):
        rel_name = md.relative_to(root).as_posix()
        targets, urls = extract(md.read_text(encoding="utf-8"))
        for t in targets:
            if is_external(t):
                ext.setdefault(t.rstrip(".,;:"), []).append(rel_name)
                continue
            problem = check_relative(md, t, root)
            if problem:
                rep.broken.append(f"{rel_name}: [{t}] {problem}")
            else:
                rep.relative_ok += 1
        for u in urls:
            ext.setdefault(u, []).append(rel_name)
    for url, files in sorted(ext.items()):
        kind = classify(url)
        where = ", ".join(sorted(set(files)))
        if kind == "private":
            rep.private.append(f"{url} ({where}): private, not checked")
        elif kind == "skip":
            rep.skipped.append(f"{url} ({where})")
        elif offline:
            rep.unchecked.append(f"{url} ({where}): not checked (offline)")
        else:
            target = probe_url(url)
            ok, status = check_external(target, head, timeout)
            if target != url:
                status = f"{status} (probed {target})"
            if ok:
                rep.ok_urls += 1
            else:
                rep.failed.append((url, status, sorted(set(files))))
    return rep


def print_report(rep: Report) -> None:
    for line in rep.broken:
        print(f"BROKEN  {line}")
    for url, status, files in rep.failed:
        print(f"FAILED  {url} -> {status} ({', '.join(files)})")
    for line in rep.private:
        print(f"PRIVATE {line}")
    for line in rep.unchecked:
        print(f"OFFLINE {line}")
    print(
        f"relative ok {rep.relative_ok}, broken {len(rep.broken)}; external ok {rep.ok_urls}, "
        f"failed {len(rep.failed)}, private (not checked) {len(rep.private)}, "
        f"skipped placeholders {len(rep.skipped)}, offline-unchecked {len(rep.unchecked)}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--offline", action="store_true", help="skip every external request")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args(argv)
    rep = run(a.root.resolve(), offline=a.offline, timeout=a.timeout)
    print_report(rep)
    return 0 if rep.clean else 1


if __name__ == "__main__":
    sys.exit(main())
