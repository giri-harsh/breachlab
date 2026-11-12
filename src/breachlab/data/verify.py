"""PRD 8.2 target verification for a built dataset (train split)."""

from __future__ import annotations

from dataclasses import dataclass

from breachlab.data.schema import Row


@dataclass(frozen=True)
class TargetReport:
    train_rows: int
    unsafe_share: float
    hard_benign_share_of_safe: float
    hi_hinglish_share_of_unsafe: float
    failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def verify_targets(rows: list[Row], *, approx_train: int = 20000, tol: float = 0.2) -> TargetReport:
    """Targets: ~20k train rows (+-20%), ~50/50 safe/unsafe (40-60%), >=15% hard benign within safe,
    >=10% hi/hinglish within unsafe."""
    train = [r for r in rows if r.split == "train"]
    safe = [r for r in train if r.label == "SAFE"]
    unsafe = [r for r in train if r.label == "UNSAFE"]
    n = len(train)
    us = len(unsafe) / n if n else 0.0
    hb = sum(r.attack_family == "hard_benign" for r in safe) / len(safe) if safe else 0.0
    hi = sum(r.language in ("hi", "hinglish") for r in unsafe) / len(unsafe) if unsafe else 0.0
    fails: list[str] = []
    if abs(n - approx_train) > tol * approx_train:
        fails.append(f"train rows {n} not within {int(tol * 100)}% of {approx_train}")
    if not 0.4 <= us <= 0.6:
        fails.append(f"unsafe share {us:.2f} outside 0.40-0.60")
    if hb < 0.15:
        fails.append(f"hard benign {hb:.3f} of safe < 0.15")
    if hi < 0.10:
        fails.append(f"hi/hinglish {hi:.3f} of unsafe < 0.10")
    return TargetReport(n, us, hb, hi, fails)
