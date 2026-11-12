"""Nickname moderation (FR-4): shape check + blocklist + optional LLM check."""

from __future__ import annotations

import re
from collections.abc import Callable

NICK_RE = re.compile(r"^[A-Za-z0-9_\- .]{3,20}$")
BLOCKLIST = {
    "fuck", "shit", "bitch", "cunt", "nigg", "faggot", "rape", "nazi", "hitler", "slut", "whore",
    "chutiya", "madarchod", "bhenchod", "gandu", "randi", "admin", "moderator",
}  # fmt: skip
LEET = str.maketrans("013457@$", "oleastas")


def normalise(nick: str) -> str:
    return re.sub(r"[^a-z]", "", nick.lower().translate(LEET))


def check_nickname(nick: str, llm_check: Callable[[str], bool] | None = None) -> tuple[bool, str]:
    """Return (ok, reason). ``llm_check(nick)`` returns True if the nickname is acceptable."""
    nick = nick.strip()
    if not NICK_RE.match(nick):
        return False, "3-20 characters: letters, digits, space, . _ -"
    flat = normalise(nick)
    if any(b in flat for b in BLOCKLIST):
        return False, "nickname not allowed"
    if llm_check is not None:
        try:
            if not llm_check(nick):
                return False, "nickname not allowed"
        except Exception:  # noqa: BLE001 - moderation outage must not block play; blocklist still applied
            pass
    return True, "ok"
