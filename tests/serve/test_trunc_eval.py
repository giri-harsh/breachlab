from pathlib import Path
from typing import Any

import numpy as np
import pytest

from breachlab.data.schema import Row
from breachlab.serve import trunc_eval as te
from breachlab.train.prompts import prompt_parts

WORDS = ["safe", "inject", "persona", "extract"]


def enc(s: str) -> list[int]:
    return [ord(c) for c in s]


def make_logits_fn(calls: list[int]) -> Any:
    """Attack iff the word 'EVIL' survives in the message part; counts forward passes."""

    def fn(ids: list[int]) -> Any:
        calls.append(len(ids))
        text = "".join(chr(i) for i in ids).split("<|im_start|>user\n")[-1]
        out = np.zeros(4)
        out[1 if "EVIL" in text else 0] = 5.0
        return out

    return fn


def rows() -> list[Row]:
    return [
        Row.new("hello there", "SAFE", "none", "t"),
        Row.new("x" * 300, "SAFE", "none", "t"),
        Row.new("EVIL " + "y" * 20, "UNSAFE", "injection", "t"),
        Row.new("z" * 300 + "EVIL", "UNSAFE", "injection", "t"),  # tail attack
        Row.new("q" * 100 + "EVIL" + "q" * 100, "UNSAFE", "injection", "t"),  # middle attack
    ]


def test_score_rungs_caches_identical_prompts() -> None:
    calls: list[int] = []
    texts = [r.text for r in rows()]
    out = te.score_rungs(texts, enc, make_logits_fn(calls), WORDS, rungs=(512, 256, 64))
    assert set(out) == {512, 256, 64} and out[512].shape == (5, 4)
    # nothing exceeds 256 chars except the 300-char rows: caps 512 and 256 share identical ids for short rows
    assert np.array_equal(out[512][0], out[256][0]) and np.array_equal(out[512][0], out[64][0])
    assert len(calls) < 3 * len(texts)


def test_truncation_keeps_head_and_tail_but_loses_the_middle() -> None:
    texts = [r.text for r in rows()]
    out = te.score_rungs(texts, enc, make_logits_fn([]), WORDS, rungs=(512, 64))
    wide, narrow = out[512].argmax(axis=1), out[64].argmax(axis=1)
    assert wide.tolist() == [0, 0, 1, 1, 1]
    assert narrow.tolist() == [0, 0, 1, 1, 0]  # the middle attack is cut out at 64 tokens


def test_on_rung_callback_reports_forward_passes() -> None:
    seen: list[tuple[int, float]] = []
    te.score_rungs(
        ["abc"],
        enc,
        make_logits_fn([]),
        WORDS,
        rungs=(512, 256),
        on_rung=lambda cap, lg, info: seen.append((cap, info["fresh_forward_passes"])),
    )
    assert seen == [(512, 1.0), (256, 0.0)]  # the second cap hits the cache


def test_truncation_stats() -> None:
    st = te.truncation_stats(["a" * 10, "b" * 300], enc, rungs=(512, 100))
    assert st[512]["truncated_share"] == 0.0 and st[100]["truncated_share"] == 0.5
    assert st[100]["mean_message_tokens"] == pytest.approx((10 + 100) / 2)
    assert st[100]["max_message_tokens"] == 300


def test_parity_table_rule() -> None:
    table = {
        512: {"auroc": 0.990, "fpr": 0.020, "asr": 0.05},
        256: {"auroc": 0.9885, "fpr": 0.024, "asr": 0.05},  # -0.0015 and +0.4 pp: parity
        128: {"auroc": 0.9870, "fpr": 0.020, "asr": 0.05},  # AUROC drop 0.003: no parity
        64: {"auroc": 0.990, "fpr": 0.030, "asr": 0.07},  # FPR +1.0 pp: no parity
    }
    p = te.parity_table(table)
    assert [p[c]["keeps_parity"] for c in (512, 256, 128, 64)] == [True, True, False, False]
    assert p[64]["d_fpr_pp"] == pytest.approx(1.0) and p[64]["d_asr_pp"] == pytest.approx(2.0)


def test_auroc_matches_reference_implementation() -> None:
    from breachlab.eval.metrics import auroc
    from breachlab.train.calibrate import to_preds

    rs = rows()
    lg = np.random.default_rng(1).normal(size=(5, 4))
    y = np.array([int(r.label == "UNSAFE") for r in rs])
    p = te._p_unsafe(lg, 1.3)
    ref = auroc(to_preds(rs, lg, 1.3, 0.5))
    assert te._auroc(y, p) == pytest.approx(ref)


def test_paired_bootstrap_identical_logits_gives_zero_interval() -> None:
    rs = rows() * 20
    lg = np.random.default_rng(0).normal(size=(len(rs), 4))
    ci = te.paired_bootstrap(rs, lg, lg.copy(), 1.0, 0.3, n_boot=20)
    assert all(v == [0.0, 0.0] for v in ci.values())


def test_evaluate_report_shape(tmp_path: Path) -> None:
    rep = te.evaluate(
        rows() * 4,
        enc,
        make_logits_fn([]),
        WORDS,
        1.0,
        0.5,
        modes=("clip500", "full"),
        rungs=(512, 64),
        out_dir=tmp_path,
        n_boot=10,
    )
    assert rep["n_rows"] == 20 and rep["n_attacks"] == 12
    for mode in ("clip500", "full"):
        m = rep["modes"][mode]
        assert set(m["metrics"]) == {512, 64} and m["vs_512"][512]["keeps_parity"] is True
        assert m["identical_logits_vs_512"][512] is True
        assert set(m["bootstrap_vs_512"]) == {64}
    # the middle attack is lost at 64 tokens: ASR rises, so the cap is not a free lunch
    assert rep["modes"]["full"]["metrics"][64]["asr"] > rep["modes"]["full"]["metrics"][512]["asr"]
    assert (tmp_path / "truncation_quality.json").exists()
    assert (tmp_path / "trunc_clip500_64_logits.npy").exists()


def test_read_gold_and_script(tmp_path: Path) -> None:
    import json

    from breachlab.serve import trunc_job

    f = tmp_path / "g.jsonl"
    f.write_text("\n".join(json.dumps(r.model_dump()) for r in rows()) + "\n\n", "utf-8")
    assert len(te.read_gold(f)) == 5
    script = trunc_job.trunc_script(["--n-boot", "5"])
    compile(script, "run.py", "exec")
    assert "trunc_eval" in script and "onnxruntime==1.31.1" in script
    assert prompt_parts(WORDS)[0].startswith("<|im_start|>system")
