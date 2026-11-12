"""commit-msg hook: reject AI attribution and non-conventional messages."""

import re
import sys
from pathlib import Path

BANNED = ("claude", "anthropic", "co-authored-by", "generated with", "\U0001f916")
CONVENTIONAL = re.compile(
    r"^(feat|fix|chore|docs|test|refactor|ci|build|perf|style|revert)(\([\w\-/]+\))?!?: .+"
)


def check(message: str) -> str | None:
    lowered = message.lower()
    for word in BANNED:
        if word in lowered:
            return f"commit message contains banned text: {word!r}"
    first = message.strip().splitlines()[0] if message.strip() else ""
    if not first.startswith("Merge ") and not CONVENTIONAL.match(first):
        return "commit message must follow Conventional Commits"
    return None


def main() -> int:
    err = check(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
