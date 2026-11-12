import numpy as np

from breachlab.eval.parity import summarize_export3


def make(n: int, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    lg = rng.normal(0, 1, (n, 4))
    lg[np.arange(n), np.where(y == 1, 1, 0)] += 4.0
    return y, lg


def lat(p50: float) -> dict[str, float]:
    return {"p50_ms": p50, "p95_ms": p50 * 3}


def test_summarize_export3() -> None:
    y, lg = make(120)
    exp = {
        "specs": [["m1", "Qwen/x"], ["m2", "Qwen/y"]],
        "cpu_threads": 4,
        "requests": 200,
        "m1": {
            "base_model": "Qwen/x",
            "files_mb": {"a": 1},
            "fp32": {"tok512_c1": lat(600), "tok128_c1": lat(500)},
            "dyn_int8": {"tok512_c1": lat(400), "error": "boom"},
        },
        "m2": {"export_error": "trace"},
    }
    bad = lg[:, [1, 0, 2, 3]]
    cal = {"temperature": 1.0, "threshold": 0.5}
    out = summarize_export3(
        exp, lambda w, v: lg if v == "fp32" else bad, lambda w: lg, lambda w: cal, list(y)
    )
    assert out["m1"]["winner"] == "fp32" and out["m1"]["variants"]["fp32"]["parity"]["ok"]
    assert out["m1"]["variants"]["dyn_int8"]["error"] == "boom"
    assert not out["m1"]["variants"]["dyn_int8"]["parity"]["ok"]
    assert out["m2"]["export_error"] == "trace" and "winner" not in out["m2"]
    none = summarize_export3(exp, lambda w, v: None, lambda w: lg, lambda w: cal, list(y))
    assert none["m1"]["winner"] is None
