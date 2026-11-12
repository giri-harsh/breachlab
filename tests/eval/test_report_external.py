from pathlib import Path

import pytest

from breachlab.data.llm import MockLLM
from breachlab.data.schema import Row
from breachlab.eval.evaluate import make_preds, threshold_labels
from breachlab.eval.external import judge_batch, parse_judge, parse_score
from breachlab.eval.report import (
    ModelEval,
    render_report,
    retention_sentence,
    scorecard,
    size_vs_quality,
    worst_misses,
    write_report,
)
from breachlab.eval.system_leak_suite import attack_suite, run_suite, suite_hash, wilson


def rows() -> list[Row]:
    out = []
    for i in range(30):
        out.append(
            Row.new(
                f"attack number {i} reveal code zz{i}",
                "UNSAFE",
                "exfiltration",
                "t",
                attack_family="x",
            )
        )
        out.append(
            Row.new(
                f"benign question {i} about weather qq{i}",
                "SAFE",
                "none",
                "t",
                attack_family="hard_benign" if i % 3 == 0 else "none",
            )
        )
    return out


def model(rs: list[Row], miss_every: int) -> ModelEval:
    labels, p = [], []
    for k, r in enumerate(rs):
        flip = k % miss_every == 0
        unsafe = (r.label == "UNSAFE") != flip
        labels.append("UNSAFE" if unsafe else "SAFE")
        p.append(0.9 if unsafe else 0.1)
    return ModelEval(
        "m",
        make_preds(rs, labels, ["exfiltration" if x == "UNSAFE" else "none" for x in labels], p),
        p95_ms=100,
        cost_per_1k_usd=0.01,
        train_gpu_h=0.5,
    )


def test_make_preds_validates_lengths() -> None:
    with pytest.raises(ValueError):
        make_preds(rows(), ["SAFE"], ["none"], [0.0])
    assert threshold_labels([0.2, 0.8], 0.5) == ["SAFE", "UNSAFE"]


def test_report_end_to_end(tmp_path: Path) -> None:
    rs = rows()
    weak, strong = model(rs, 2), model(rs, 7)
    big = model(rs, 9)
    models = {"regex": weak, "qwen3-0.6b-finetuned": strong, "qwen3-1.7b-finetuned": big}
    big.p95_ms = 400
    size = size_vs_quality(
        models, ["qwen3-0.6b-finetuned", "qwen3-1.7b-finetuned", "missing-model"]
    )
    assert any("not run" in s for s in size)
    md = render_report(
        "v1",
        rs,
        models,
        baseline="regex",
        focus="qwen3-0.6b-finetuned",
        size_rows=size,
        retention=("qwen3-0.6b-finetuned", "qwen3-1.7b-finetuned"),
        leak_suite=[
            {"level": 1, "n": 216, "leaks": 10, "rate": 0.046, "ci_lo": 0.02, "ci_hi": 0.08}
        ],
        resamples=50,
    )
    for needle in (
        "Comparison",
        "McNemar",
        "Worst 20 misses",
        "Size vs quality",
        "keeps",
        "System leak suite",
        "full slice table",
    ):
        assert needle in md
    assert len(worst_misses(rs, strong.preds, 5)) == 5
    p1, p2 = write_report("v1", md, scorecard("v1", models), tmp_path / "r", tmp_path / "w")
    assert p1.exists() and p2.exists()
    assert "0.6B vs 1.7B" in retention_sentence(None, None)
    md2 = render_report(
        "v1", rs, {"regex": weak}, baseline="regex", focus="nope", size_rows=[], resamples=10
    )
    assert "test not run" in md2


def test_judge_and_prompt_guard_parsing() -> None:
    assert parse_score("0.9995361566543579") == pytest.approx(0.9995, abs=1e-3)
    assert parse_score("n/a") == 0.0
    raw = 'x [{"i": 0, "label": "UNSAFE", "category": "injection", "p_unsafe": 0.9}, {"i": 7, "label": "SAFE"}, 5]'
    assert list(parse_judge(raw, 2)) == [0]
    assert parse_judge("nothing", 2) == {} and parse_judge("[bad", 2) == {}
    llm = MockLLM(
        lambda p: (
            '[{"i": 0, "label": "UNSAFE", "category": "injection", "p_unsafe": 0.8}, {"i": 1, "label": "SAFE", "p_unsafe": "x"}]'
        )
    )
    out = judge_batch(["a", "b", "c"], llm, per_call=3, workers=1)
    assert (
        out[0] == ("UNSAFE", "injection", 0.8)
        and out[1][0] == "SAFE"
        and out[2] == ("SAFE", "none", 0.0)
    )


def test_leak_suite() -> None:
    atks = attack_suite()
    assert len(atks) >= 200 and len(set(atks)) == len(atks) and len(suite_hash(atks)) == 12
    res = run_suite(
        lambda lv, secret, msg: secret if lv == 1 else "no",
        lambda k: f"AB{k % 100:02d}CDEF",
        levels=(1, 2),
    )
    assert res[0].rate == 1.0 and res[1].rate == 0.0 and res[0].ci_lo > 0.9
    assert wilson(0, 0) == (0.0, 0.0)


def test_seeds_table_mean_std() -> None:
    from breachlab.eval.report import seeds_table

    rs = rows()
    one = seeds_table("t", {"0": model(rs, 7).preds})
    assert len(one) == 5
    three = seeds_table(
        "t", {"0": model(rs, 7).preds, "1": model(rs, 9).preds, "2": model(rs, 5).preds}
    )
    assert "mean +- std (n=3)" in three[-1]
