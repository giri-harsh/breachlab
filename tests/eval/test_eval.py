import math

import numpy as np
import pytest

from breachlab.eval import metrics as m
from breachlab.eval.bootstrap import bootstrap_ci
from breachlab.eval.calibration import ece, ece_from_logits, fit_temperature, softmax
from breachlab.eval.gate import (
    CandidateResult,
    ChampionResult,
    GateConfig,
    evaluate_gate,
    load_gate,
)
from breachlab.eval.mcnemar import mcnemar


def P(
    y: int, yp: int, p: float, ct: str = "none", cp: str = "none", hb: bool = False, **sl: str
) -> m.Pred:
    return m.Pred(y, yp, p, ct, cp, hb, tuple(sl.items()))


PREDS = [
    P(1, 1, 0.9, "injection", "injection", family="a"),
    P(1, 1, 0.8, "jailbreak", "injection", family="a"),
    P(1, 0, 0.3, "injection", "none", family="b"),
    P(1, 0, 0.2, "exfiltration", "none", family="b"),
    P(0, 0, 0.1),
    P(0, 0, 0.2, hb=True),
    P(0, 1, 0.7, hb=True),
    P(0, 0, 0.05),
]


def test_core_metrics() -> None:
    assert m.asr(PREDS) == 0.5
    assert m.fpr(PREDS) == 0.25
    assert m.fpr(PREDS, hard_only=True) == 0.5
    prec, rec, f1 = m.prf(PREDS)
    assert prec == pytest.approx(2 / 3) and rec == 0.5 and f1 == pytest.approx(4 / 7)
    assert m.category_accuracy(PREDS) == 0.5
    assert m.confusion(PREDS) == {"tp": 2, "fp": 1, "fn": 2, "tn": 3}
    s = m.summarize(PREDS)
    assert set(s) >= {"asr", "fpr", "f1", "auroc", "hard_benign_fpr"}
    assert set(m.per_category(PREDS)) == {"injection", "jailbreak", "exfiltration"}


def test_auroc_known_values() -> None:
    perfect = [P(1, 1, 0.9), P(1, 1, 0.8), P(0, 0, 0.2), P(0, 0, 0.1)]
    assert m.auroc(perfect) == 1.0
    assert m.auroc([P(1, 1, 0.5), P(0, 1, 0.5)]) == 0.5  # ties
    assert m.auroc([P(1, 1, 0.5)]) == 0.5  # single class
    inverted = [P(1, 0, 0.1), P(0, 1, 0.9)]
    assert m.auroc(inverted) == 0.0


def test_empty_edge_cases() -> None:
    assert m.asr([]) == 0.0 and m.fpr([]) == 0.0 and m.prf([]) == (0.0, 0.0, 0.0)


def test_slices_and_worst() -> None:
    rep = m.slice_report(PREDS)
    assert rep["family"]["b"]["asr"] == 1.0 and rep["family"]["a"]["asr"] == 0.0
    assert m.worst_slice(PREDS, min_n=2) == ("family", "b", 1.0)
    assert m.worst_slice(PREDS, min_n=50) is None


def test_threshold_for_fpr() -> None:
    rng = np.random.default_rng(0)
    preds = [P(0, 0, float(x)) for x in rng.random(1000)] + [P(1, 1, 0.99)]
    thr = m.threshold_for_fpr(preds, 0.02)
    blocked = sum(p.p_unsafe >= thr for p in preds if p.y_true == 0)
    assert blocked <= 20
    assert m.threshold_for_fpr([P(1, 1, 0.5)]) == 0.5


def test_bootstrap_seeded_and_covers() -> None:
    a = bootstrap_ci(PREDS, m.asr, resamples=200, seed=1)
    b = bootstrap_ci(PREDS, m.asr, resamples=200, seed=1)
    assert a == b and a[1] <= a[0] <= a[2]
    assert bootstrap_ci([], m.asr) == (0.0, 0.0, 0.0)
    big = [P(1, int(i % 10 == 0), 0.5) for i in range(1000)]  # asr 0.9
    pt, lo, hi = bootstrap_ci(big, m.asr, resamples=1000, seed=3)
    assert lo < 0.9 < hi and hi - lo < 0.05


def test_mcnemar_exact_and_chi2() -> None:
    a = [True] * 10 + [False] * 0
    b = [False] * 10
    r = mcnemar(a, b)
    assert (r.b, r.c, r.method) == (10, 0, "exact")
    assert r.p_value == pytest.approx(2 / 2**10)
    assert mcnemar([True, False], [True, False]).method == "none"
    big_a = [True] * 60 + [False] * 20
    big_b = [False] * 60 + [True] * 20
    r2 = mcnemar(big_a, big_b)
    assert r2.method == "chi2" and r2.p_value < 1e-4
    expected = math.erfc(math.sqrt((abs(60 - 20) - 1) ** 2 / 80 / 2))
    assert r2.p_value == pytest.approx(expected)
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_ece_and_temperature() -> None:
    conf = np.array([1.0] * 10)
    assert ece(conf, np.array([1.0] * 5 + [0.0] * 5)) == pytest.approx(0.5)
    assert ece(np.array([]), np.array([])) == 0.0
    rng = np.random.default_rng(0)
    n = 4000
    labels = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4))
    logits[np.arange(n), labels] += 2.0
    over = logits * 4  # overconfident model
    t = fit_temperature(over, labels)
    assert 1.5 < t < 8.0
    assert ece_from_logits(over, labels, t) < ece_from_logits(over, labels, 1.0)
    assert softmax(np.array([[0.0, 0.0]])).tolist() == [[0.5, 0.5]]


GOOD = CandidateResult(
    asr=0.1,
    fpr=0.01,
    p95_latency_ms=200,
    ece=0.03,
    mcnemar_p=0.001,
    fresh_asr=0.1,
    slice_asr={"a": 0.1},
)
CHAMP = ChampionResult(asr=0.2, fpr=0.012, fresh_asr=0.15, slice_asr={"a": 0.08})


def test_gate_pass_and_each_failure() -> None:
    cfg = load_gate().model_copy(
        update={"p95_latency_ms_max": 300.0}
    )  # independent of the shipped threshold
    assert evaluate_gate(GOOD, CHAMP, cfg).passed
    assert not evaluate_gate(GOOD.model_copy(update={"mcnemar_p": 0.2}), CHAMP, cfg).checks[
        "gold_asr"
    ][0]
    assert not evaluate_gate(GOOD.model_copy(update={"asr": 0.25}), CHAMP, cfg).passed
    assert not evaluate_gate(GOOD.model_copy(update={"mcnemar_p": None}), CHAMP, cfg).passed
    assert not evaluate_gate(GOOD.model_copy(update={"fpr": 0.025}), CHAMP, cfg).checks["fpr"][0]
    assert not evaluate_gate(
        GOOD.model_copy(update={"fpr": 0.0185}), CHAMP.model_copy(update={"fpr": 0.012}), cfg
    ).checks["fpr"][0]
    assert not evaluate_gate(GOOD.model_copy(update={"p95_latency_ms": 301}), CHAMP, cfg).checks[
        "latency"
    ][0]
    assert not evaluate_gate(GOOD.model_copy(update={"slice_asr": {"a": 0.2}}), CHAMP, cfg).checks[
        "slices"
    ][0]
    assert not evaluate_gate(GOOD.model_copy(update={"ece": 0.06}), CHAMP, cfg).checks["ece"][0]
    assert not evaluate_gate(GOOD.model_copy(update={"fresh_asr": 0.3}), CHAMP, cfg).checks[
        "fresh_asr"
    ][0]
    v1 = evaluate_gate(GOOD, ChampionResult(asr=0.2, fpr=0.012), GateConfig())
    assert v1.checks["fresh_asr"][0] and v1.passed


def test_gate_base_champion_skips_fpr_delta() -> None:
    from breachlab.eval.gate import CandidateResult, ChampionResult, evaluate_gate

    cand = CandidateResult(asr=0.01, fpr=0.015, p95_latency_ms=200, ece=0.01, mcnemar_p=0.001)
    assert not evaluate_gate(cand, ChampionResult(asr=1.0, fpr=0.0)).checks["fpr"][0]
    assert evaluate_gate(cand, ChampionResult(asr=1.0, fpr=0.0, is_base=True)).passed
    assert not evaluate_gate(
        cand.model_copy(update={"fpr": 0.03}), ChampionResult(asr=1.0, fpr=0.0, is_base=True)
    ).passed
