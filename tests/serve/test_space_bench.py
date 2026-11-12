from pathlib import Path

import numpy as np
import pytest

from breachlab.data.schema import Row
from breachlab.serve import space_bench


def test_clip_and_bench_counts_errors() -> None:
    assert space_bench.clip(["a" * 600, "b"]) == ["a" * 500, "b"]
    calls = {"n": 0}

    def post(text: str) -> dict[str, float]:
        calls["n"] += 1
        if text == "bad":
            raise RuntimeError("boom")
        return {"latency_ms": 2.0}

    res = space_bench.bench(post, ["ok", "bad"], requests=40, concurrency=1, warmup=0)
    assert res.n == 40 and 0 < res.errors < 40 and res.model_p50_ms == 2.0
    assert res.rtt_p95_ms >= res.rtt_p50_ms >= 0 and calls["n"] == 40
    res4 = space_bench.bench(
        lambda t: {"latency_ms": 1.0}, ["x"], requests=20, concurrency=4, warmup=2
    )
    assert res4.concurrency == 4 and res4.errors == 0 and res4.model_mean_ms == 1.0


def test_bench_all_failures_raises() -> None:
    def post(text: str) -> dict[str, float]:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        space_bench.bench(post, ["x"], requests=5, warmup=0)


def rows() -> list[Row]:
    out = [Row.new(f"safe {i}", "SAFE", "none", "t") for i in range(6)]
    out += [Row.new(f"attack {i}", "UNSAFE", "injection", "t") for i in range(6)]
    return out


def logits_for(rs: list[Row], noise: float, seed: int = 0) -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    out = np.zeros((len(rs), 4))
    for i, r in enumerate(rs):
        out[i, 0 if r.label == "SAFE" else 1] = 5.0
    return out + rng.normal(0, noise, out.shape)


def test_parity_identical_and_degraded() -> None:
    rs = rows()
    t = logits_for(rs, 0.0)
    same = space_bench.parity(rs, t, t.copy(), 1.0, 0.5)
    assert same["auroc_delta"] == 0 and same["label_agreement"] == 1.0 and same["fpr_delta_pp"] == 0
    assert same["max_abs_logit_diff"] == 0 and same["n"] == 12
    flipped = t.copy()
    flipped[:, [0, 1]] = flipped[:, [1, 0]]  # a broken export: classes swapped
    bad = space_bench.parity(rs, t, flipped, 1.0, 0.5)
    assert bad["label_agreement"] == 0 and bad["auroc_onnx"] < bad["auroc_torch"]
    with pytest.raises(ValueError):
        space_bench.parity(rs, t, t[:3], 1.0, 0.5)


def test_read_gold(tmp_path: Path) -> None:
    f = tmp_path / "g.jsonl"
    f.write_text("\n".join(r.model_dump_json() for r in rows()[:2]) + "\n\n")
    assert [r.text for r in space_bench.read_gold(f)] == ["safe 0", "safe 1"]
