"""Pure parts of the int4 / int8 guard export (ADR-0080): variant table, bootstrap intervals, acceptance rule, graph
helpers on stand-in graph objects, and the kernel script. The ONNX rewrite itself runs only in the Kaggle kernel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from breachlab.serve import quant_export as q
from breachlab.serve import quant_job


def init(name: str, *dims: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, dims=list(dims))


def node(op: str, name: str, *inputs: str) -> SimpleNamespace:
    return SimpleNamespace(op_type=op, name=name, input=list(inputs), output=[name + "_out"])


def test_variant_table_covers_the_lane_matrix() -> None:
    names = [v.name for v in q.VARIANTS]
    assert len(set(names)) == len(names)
    bits_blocks = {
        (v.bits, v.block) for v in q.VARIANTS if v.embed == "int8" and v.accuracy_level is None
    }
    assert {(4, 32), (4, 64), (8, 32)} <= bits_blocks
    assert "q4b32" in [v.name for v in q.VARIANTS if v.ship]
    assert [v.name for v in q.select(["q8b32", "q4b32"])] == ["q8b32", "q4b32"]
    with pytest.raises(KeyError, match="nope"):
        q.select(["q4b32", "nope"])


def test_rate_ci_brackets_the_rate_and_is_deterministic() -> None:
    flags = np.array([True] * 20 + [False] * 180)
    rate, lo, hi = q.rate_ci(flags)
    assert rate == pytest.approx(0.1) and lo < rate < hi and lo > 0.05 and hi < 0.16
    assert q.rate_ci(flags) == (rate, lo, hi)
    assert np.isnan(q.rate_ci(np.array([], dtype=bool))[0])


def test_judge_accepts_inside_interval_and_rejects_each_violation() -> None:
    ci = (0.02, 0.01, 0.03)
    ok = q.judge(0.9990, 0.9985, 0.025, 0.02, ci, ci, True)
    assert ok["pass"] and ok["auroc_drop"] == pytest.approx(0.0005)
    assert not q.judge(0.9990, 0.9960, 0.02, 0.02, ci, ci, True)[
        "pass"
    ]  # AUROC dropped 0.003 > 0.002
    assert not q.judge(0.9990, 0.9990, 0.05, 0.02, ci, ci, True)[
        "pass"
    ]  # FPR outside the fp32 interval
    assert not q.judge(0.9990, 0.9990, 0.02, 0.0, ci, ci, True)[
        "pass"
    ]  # ASR outside the fp32 interval
    edge = q.judge(0.9990, 0.9970, 0.02, 0.02, ci, ci, False)
    assert (
        edge["pass"] and not edge["adr0057_ok"]
    )  # the lane rule and ADR-0057 are reported separately


def test_embedding_and_head_detection() -> None:
    g = SimpleNamespace(
        initializer=[
            init("emb", 151936, 1024),
            init("w_q", 1024, 2048),
            init("head", 1024, 4),
            init("cos", 4096, 128),
        ],
        node=[
            node("Gather", "emb_gather", "emb", "ids"),
            node("Gather", "rope_gather", "cos", "pos"),
            node("MatMul", "q_proj", "x", "w_q"),
            node("MatMul", "head_mm", "h", "head"),
            node("MatMul", "scores", "q", "k"),  # no constant weight: not a candidate
        ],
    )
    assert q.find_embedding(g).name == "emb_gather"
    assert q.head_nodes(g) == ["head_mm"]
    g.node.pop(0)
    with pytest.raises(LookupError):
        q.find_embedding(g)


def test_kernel_script_installs_pinned_runtime_and_runs_main() -> None:
    script = quant_job.quant_script(["--rows", "24"])
    assert "onnxruntime==1.31.1" in script and "onnx-ir" in script
    assert "breachlab.serve.quant_export import main" in script and "'--rows', '24'" in script
    assert "PYTHONPATH" in script  # the pinned probe subprocesses must import breachlab
    compile(script, "run.py", "exec")


def test_percentile_helper_handles_empty() -> None:
    assert np.isnan(q.pct([], 50)) and q.pct([1.0, 2.0, 3.0], 50) == 2.0
