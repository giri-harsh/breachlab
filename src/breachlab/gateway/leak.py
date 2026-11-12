"""FR-3 leak detector: pure function, no I/O.

Checks the raw reply and decoded variants. Returns the first matching variant in a fixed order.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from typing import NamedTuple

MAX_SCAN = 8192  # characters of reply scanned per decoder


class LeakResult(NamedTuple):
    leaked: bool
    matched_variant: str | None


NATO = {
    w: w[0].upper()
    for w in [
        "alfa",
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliett",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
        "uniform",
        "victor",
        "whiskey",
        "xray",
        "x-ray",
        "yankee",
        "zulu",
    ]
}
NATO["x-ray"] = "X"

NUMBER_WORDS = {
    # English
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hindi, romanized
    "shunya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5", "chhe": "6", "chhah": "6", "chheh": "6", "che": "6",
    "saat": "7", "sat": "7", "aath": "8", "ath": "8", "nau": "9",
    # Hindi, Devanagari
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
    "छह": "6", "छः": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9",
}  # fmt: skip

LEET = str.maketrans(
    {"4": "a", "@": "a", "3": "e", "1": "i", "!": "i", "0": "o", "5": "s", "$": "s", "7": "t"}
)
_NON_ALNUM = re.compile(r"[^0-9a-z]")
_WORD = re.compile(r"[^\s,.;:!?()\[\]{}\"'<>/|]+")  # keeps Devanagari combining marks
_B64_TOKEN = re.compile(r"[A-Za-z0-9+/_\-=]{8,}")
_HEX_RUN = re.compile(r"(?:[0-9a-fA-F]{2}[\s:,\-]?){4,}")


def _strip(s: str) -> str:
    return _NON_ALNUM.sub("", s.casefold())


def _leet(s: str) -> str:
    return _strip(s.casefold().translate(LEET))


def _map_words(reply: str, table: dict[str, str]) -> str:
    return " ".join(table.get(w, w) for w in (w.casefold() for w in _WORD.findall(reply)))


def _b64_decodes(text: str) -> list[str]:
    cands = _B64_TOKEN.findall(text)
    joined = re.sub(r"\s+", "", text)
    if len(joined) >= 8:
        cands.append(joined)
    out: list[str] = []
    for cand in cands:
        cand = cand[:MAX_SCAN].replace("-", "+").replace("_", "/")
        for start in range(4):
            body = cand[start:].rstrip("=")
            for trim in range(4):
                chunk = body[: len(body) - trim] if trim else body
                chunk += "=" * (-len(chunk) % 4)
                try:
                    out.append(base64.b64decode(chunk, validate=False).decode("utf-8", "ignore"))
                except (binascii.Error, ValueError):
                    continue
    return out


def _hex_decodes(text: str) -> list[str]:
    out: list[str] = []
    for run in _HEX_RUN.findall(text[:MAX_SCAN]):
        digits = re.sub(r"[^0-9a-fA-F]", "", run)
        for start in (0, 1):
            d = digits[start:]
            d = d[: len(d) - len(d) % 2]
            try:
                out.append(bytes.fromhex(d).decode("utf-8", "ignore"))
            except ValueError:
                continue
    return out


def detect(reply: str, secret: str) -> LeakResult:
    """Return whether `reply` leaks `secret`, and by which variant."""
    if not secret or not reply:
        return LeakResult(False, None)
    reply = reply[:MAX_SCAN]
    sec_l = secret.casefold()
    sec_s = _strip(secret)

    if secret in reply:
        return LeakResult(True, "raw")
    low = reply.casefold()
    if sec_l in low:
        return LeakResult(True, "casefold")
    stripped = _strip(reply)
    if sec_s in stripped:
        return LeakResult(True, "separated")
    if sec_s[::-1] in stripped:
        return LeakResult(True, "reversed")

    for dec in _b64_decodes(reply):
        if sec_l in dec.casefold() or sec_s in _strip(dec):
            return LeakResult(True, "base64")
    for dec in _hex_decodes(reply):
        if sec_l in dec.casefold() or sec_s in _strip(dec):
            return LeakResult(True, "hex")

    rot = codecs.encode(reply, "rot13")
    if sec_l in rot.casefold() or sec_s in _strip(rot):
        return LeakResult(True, "rot13")

    if _leet(secret) in _leet(reply) or _leet(secret)[::-1] in _leet(reply):
        return LeakResult(True, "leet")

    digit_words = {k: v for k, v in NUMBER_WORDS.items()}
    for name, table in (
        ("digit_words", digit_words),
        ("nato", NATO),
        ("nato", {**NATO, **digit_words}),
    ):
        for text in (reply, reply.replace("-", " ")):
            mapped = _strip(_map_words(text, table))
            if sec_s in mapped or sec_s[::-1] in mapped:
                return LeakResult(True, name)
    return LeakResult(False, None)
