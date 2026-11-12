"""Fail if any commit after the baseline carries an AI attribution trailer (hook and CI use the same rule).

History up to BASELINE is exempt (squash-merge trailers already on main; no force-push).
"""

from __future__ import annotations

import re
import subprocess
import sys

BASELINE = "83616d8"  # last commit before the no-trailer policy
BAD = re.compile(
    r"co-authored-by:|generated (with|by) .*claude|claude-session:|noreply@anthropic\.com",
    re.IGNORECASE,
)


def bad_lines(message: str) -> list[str]:
    return [ln for ln in message.splitlines() if BAD.search(ln)]


def main(rev_range: str | None = None) -> int:
    rng = rev_range or f"{BASELINE}..HEAD"
    out = subprocess.run(
        ["git", "log", rng, "--format=%H%x00%B%x01"], capture_output=True, text=True, check=False
    ).stdout
    failed = 0
    for entry in filter(None, out.split("\x01")):
        sha, _, body = entry.strip().partition("\x00")
        hits = bad_lines(body)
        if hits:
            failed += 1
            print(f"FAIL {sha[:8]}: {hits[0]!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
