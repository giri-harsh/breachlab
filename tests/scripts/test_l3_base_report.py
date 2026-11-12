"""Pure parts of the L3 base guard / opener slice report (ADR-0086, ADR-0087)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "l3_base_and_openers_report.py"
spec = importlib.util.spec_from_file_location("l3_report", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["l3_report"] = mod
spec.loader.exec_module(mod)


def logits(p_unsafe: list[float]) -> np.ndarray:
    """4-way logits whose temperature-1 p_unsafe equals the given values (safe vs the rest)."""
    out = []
    for p in p_unsafe:
        out.append([np.log(1 - p), np.log(p / 3), np.log(p / 3), np.log(p / 3)])
    return np.array(out)


def test_fpr_ci_counts_flags_among_safe_rows() -> None:
    p = np.array([0.9, 0.1, 0.2, 0.05, 0.8, 0.01, 0.02, 0.03, 0.04, 0.06])
    r = mod.fpr_ci(p, 0.5)
    assert r["rate"] == 0.2 and r["n"] == 10 and 0.0 <= r["lo"] <= r["rate"] <= r["hi"] <= 1.0


def test_slice_fprs_groups_by_key() -> None:
    meta = [{"language": "en"}, {"language": "en"}, {"language": "hi"}, {"language": "hi"}]
    out = mod.slice_fprs(meta, np.array([0.9, 0.1, 0.1, 0.1]), 0.5, "language")
    assert out["en"]["rate"] == 0.5 and out["hi"]["rate"] == 0.0


def test_base_verdict_uses_val_for_the_threshold_and_reports_gold() -> None:
    rng = np.random.default_rng(0)
    # a weak guard: attacks and benign overlap a lot
    val_y = np.array([0] * 500 + [1] * 500)
    gold_y = np.array([0] * 400 + [1] * 400)

    def draw(y: np.ndarray) -> np.ndarray:
        return np.clip(
            np.where(y == 1, rng.beta(2, 3, len(y)), rng.beta(1, 6, len(y))), 0.001, 0.999
        )

    v = mod.base_verdict(logits(list(draw(val_y))), val_y, logits(list(draw(gold_y))), gold_y, 1.0)
    assert 0.0 < v["threshold"] < 1.0
    assert v["val"]["fpr"] <= 0.02 + 1e-9  # the rule: val FPR at most 2%
    assert v["gold"]["n"] == 800 and v["gold"]["n_attacks"] == 400
    assert (
        0.5 < v["gold"]["auroc"] < 1.0 and v["gold"]["asr"]["rate"] > 0.2
    )  # weak guard: many attacks pass


def test_labels_of_reads_safe_and_unsafe(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_text('{"label":"SAFE"}\n{"label":"UNSAFE"}\n', "utf-8")
    assert mod.labels_of(f).tolist() == [0, 1]
