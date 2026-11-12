"""Baseline classifiers as pure logic: regex/keyword filter (PRD 9.2 row 1)."""

from __future__ import annotations

import re

from breachlab.eval.metrics import Pred

PATTERNS: dict[str, list[str]] = {
    "injection": [
        r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
        r"disregard (all |any )?(the )?(previous|prior|above|earlier)",
        r"forget (everything|all) (you|above)",
        r"new (instructions?|rules?) (from|by) (the )?(developer|admin|system)",
        r"\[?system\]?:",
        r"you are now",
        r"override (your|the) (rules|instructions|guidelines)",
    ],
    "jailbreak": [
        r"\bDAN\b",
        r"do anything now",
        r"developer mode",
        r"without (any )?(restrictions|filters|limitations)",
        r"jailbreak",
        r"pretend (to be|you are)",
        r"act as (an? )?(unrestricted|unfiltered)",
    ],
    "exfiltration": [
        r"(reveal|show|print|tell|give|repeat|spell|write)[^.\n]{0,40}(secret|password|code|system prompt|hidden instructions)",
        r"(secret|access) code",
        r"system prompt",
        r"verbatim",
        r"base64|\bhex\b|\brot13\b|leetspeak",
        r"backwards?",
    ],
}
COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in PATTERNS.items()}


def regex_classify(text: str) -> tuple[str, str]:
    """(label, category) for a message: first matching category wins."""
    for cat, pats in COMPILED.items():
        if any(p.search(text) for p in pats):
            return "UNSAFE", cat
    return "SAFE", "none"


def to_pred(y_true: int, label: str, cat: str, p_unsafe: float, **kw: object) -> Pred:
    return Pred(y_true=y_true, y_pred=int(label == "UNSAFE"), p_unsafe=p_unsafe, cat_pred=cat, **kw)  # type: ignore[arg-type]
