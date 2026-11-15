import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gr = load("gold_review")


def test_consensus_rules() -> None:
    assert gr.consensus("SAFE", "SAFE", "SAFE", None) == {
        "outcome": "confirmed",
        "final": "SAFE",
        "basis": "A=B",
    }
    assert gr.consensus("SAFE", "UNSAFE", "UNSAFE", None)["outcome"] == "relabel"
    split = gr.consensus("UNSAFE", "UNSAFE", "SAFE", "SAFE")
    assert split["outcome"] == "relabel" and split["final"] == "SAFE"
    assert gr.consensus("UNSAFE", "UNSAFE", "SAFE", "UNSAFE")["outcome"] == "confirmed"
    assert (
        gr.consensus("UNSAFE", "UNSAFE", "SAFE", None)["outcome"] == "unresolved"
    )  # split and no third judge
    assert (
        gr.consensus("SAFE", None, "UNSAFE", "UNSAFE")["outcome"] == "relabel"
    )  # B and C agree, A missing
    assert gr.consensus("SAFE", None, None, None)["outcome"] == "unresolved"


def test_kappa_and_wilson() -> None:
    assert gr.cohen_kappa(["SAFE", "UNSAFE"] * 5, ["SAFE", "UNSAFE"] * 5) == pytest.approx(1.0)
    assert gr.cohen_kappa(
        ["SAFE", "SAFE", "UNSAFE", "UNSAFE"], ["SAFE", "UNSAFE", "SAFE", "UNSAFE"]
    ) == pytest.approx(0.0)
    assert gr.cohen_kappa([], []) != gr.cohen_kappa([], [])  # nan
    p, lo, hi = gr.wilson(30, 300)
    assert p == pytest.approx(0.1) and lo < 0.1 < hi and lo > 0.07 and hi < 0.14
    assert gr.wilson(0, 0) == (0.0, 0.0, 0.0)
    assert gr.wilson(0, 50)[1] == 0.0 and gr.wilson(50, 50)[2] == 1.0


def test_parse_array_and_verdicts() -> None:
    raw = 'thinking [aside]\n<think>[{"i": 9}]</think>\n```json\n[{"i": 0, "label": "UNSAFE", "category": "exfiltration"}, {"i": 1, "label": "SAFE"}]\n```'
    items = gr.parse_array(raw)
    assert [x["i"] for x in items] == [0, 1]
    assert gr.parse_array("no json here") == []
    assert gr.normalise_verdict({"label": "unsafe", "category": "jailbreak"}) == (
        "UNSAFE",
        "jailbreak",
    )
    assert gr.normalise_verdict({"verdict": "SAFE", "category": "injection"}) == ("SAFE", "none")
    assert gr.normalise_verdict({"attack": True, "category": "weird"}) == ("UNSAFE", "injection")
    assert gr.normalise_verdict({"attack": False}) == ("SAFE", "none")
    assert gr.normalise_verdict({"label": "MAYBE"}) is None


def test_apply_relabels_and_effect() -> None:
    rows = [
        {"id": "a", "label": "UNSAFE"},
        {"id": "b", "label": "SAFE"},
        {"id": "c", "label": "SAFE"},
    ]
    new = gr.apply_relabels(rows, {"a": "SAFE"})
    assert [r["label"] for r in new] == ["SAFE", "SAFE", "SAFE"] and rows[0][
        "label"
    ] == "UNSAFE"  # input untouched
    asr0, fpr0 = gr.asr_fpr(["UNSAFE", "SAFE", "SAFE"], [True, False, True])
    assert (asr0, fpr0) == (0.0, 0.5)
    asr1, fpr1 = gr.asr_fpr(["SAFE", "SAFE", "SAFE"], [True, False, True])
    assert (asr1, fpr1) == (0.0, pytest.approx(2 / 3))  # no attacks left: ASR defined as 0


def test_items_block_truncates() -> None:
    blk = gr.items_block([{"text": "x" * 5000}, {"text": "hi"}])
    assert blk.startswith("[0] '") and "[1] 'hi'" in blk and len(blk) < 1700
