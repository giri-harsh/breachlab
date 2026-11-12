"""PII scrub (PRD 8.4): emails, phones (incl. Indian formats), URLs carrying tokens."""

from __future__ import annotations

import re

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# +91 98765 43210, 098765-43210, 9876543210, +1 (415) 555-2671, 415-555-2671
PHONE = re.compile(
    r"(?<![\w])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
    r"|(?<![\w])\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
)
TOKEN_URL = re.compile(
    r"https?://\S*(?:token|key|secret|sig|auth|password|apikey|access_token)=\S+", re.IGNORECASE
)
AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")


def scrub(text: str) -> str:
    text = TOKEN_URL.sub("[URL]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = AADHAAR.sub("[ID]", text)
    return PHONE.sub("[PHONE]", text)


def has_pii(text: str) -> bool:
    return scrub(text) != text
