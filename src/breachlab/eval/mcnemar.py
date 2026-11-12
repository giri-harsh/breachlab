"""McNemar test: exact binomial for few discordant pairs, continuity-corrected chi-square otherwise."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class McNemar:
    b: int  # A correct, B wrong
    c: int  # A wrong, B correct
    p_value: float
    method: str


def mcnemar(correct_a: list[bool], correct_b: list[bool], exact_max: int = 25) -> McNemar:
    if len(correct_a) != len(correct_b):
        raise ValueError("paired samples must have equal length")
    b = sum(x and not y for x, y in zip(correct_a, correct_b, strict=True))
    c = sum(y and not x for x, y in zip(correct_a, correct_b, strict=True))
    n = b + c
    if n == 0:
        return McNemar(b, c, 1.0, "none")
    if n <= exact_max:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
        return McNemar(b, c, p, "exact")
    stat = (abs(b - c) - 1) ** 2 / n
    return McNemar(b, c, math.erfc(math.sqrt(stat / 2)), "chi2")
