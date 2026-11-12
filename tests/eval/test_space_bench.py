import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from breachlab.data.schema import Row
from breachlab.eval.metrics import Pred
from breachlab.eval.report import ModelEval, render_report
from breachlab.eval.space_bench import (
    B2_PATH,
    PREFERRED_SOURCE,
    RUNGS,
    Rung,
    ladder_table,
    latency_section,
    load_b2,
    load_space_bench,
    load_vercel_bench,
    pick,
    quality_delta,
    repeats,
    ship_rung,
)

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def entry(p50: float, p95: float, quality: dict[str, float] | None = None) -> dict[str, object]:
    d: dict[str, object] = {
        "cold_start_s": 5.0,
        "c1": {"model_p50_ms": p50, "model_p95_ms": p95},
        "c4": {"model_p50_ms": p50 + 1, "model_p95_ms": p95 + 1},
    }
    if quality:
        d["quality"] = quality
        d["quality_n"] = 1500
    return d


def q(asr: float, fpr: float, f1: float, cat: float) -> dict[str, float]:
    return {"asr": asr, "fpr": fpr, "f1": f1, "auroc": 0.99, "category_accuracy": cat}


def write_bench(root: Path, run: str, data: dict[str, object]) -> None:
    d = root / run / "bench"
    d.mkdir(parents=True)
    (d / "bench.json").write_text(json.dumps({"cpus_total": 4, **data}))


@pytest.fixture
def bench_root(tmp_path: Path) -> Path:
    write_bench(tmp_path, "bench2", {"fp32": entry(870, 3375), "kv": entry(626, 3664)})
    write_bench(
        tmp_path,
        "bench3",
        {
            "kv": entry(442.6, 2325.78, q(0.0044, 0.0159, 0.9884, 0.8675)),
            "kv@128": entry(456, 1279, q(0.0044, 0.0159, 0.9884, 0.8557)),
        },
    )
    write_bench(tmp_path, "bench4", {"kv@64": entry(394, 622, q(0.0029, 0.0196, 0.9869, 0.8309))})
    return tmp_path


def test_load_pick_and_repeats(bench_root: Path) -> None:
    rungs = load_space_bench(bench_root)
    assert len(rungs) == 5
    kv = pick(rungs, "kv")
    assert kv is not None and kv.source == PREFERRED_SOURCE["kv"] == "bench3"
    assert ship_rung(rungs) is kv and kv.c1_p95_ms == 2325.78 and kv.c1_p50_ms == 442.6
    assert [r.source for r in repeats(rungs, "kv")] == ["bench2"]
    assert pick(rungs, "kv@96") is None
    assert pick([r for r in rungs if r.source == "bench2"], "kv").source == "bench2"  # type: ignore[union-attr]
    assert load_space_bench(bench_root / "missing") == []


def test_quality_delta_flags_loss_only_when_worse(bench_root: Path) -> None:
    rungs = load_space_bench(bench_root)
    ref = pick(rungs, "kv")
    assert quality_delta(pick(rungs, "fp32"), ref) == (False, "not measured")  # type: ignore[arg-type]
    assert quality_delta(ref, ref) == (False, "reference")  # type: ignore[arg-type]
    worse, text = quality_delta(pick(rungs, "kv@128"), ref)  # type: ignore[arg-type]
    assert worse and "cat acc -1.18 pp" in text
    better = Rung("x", "s", 1, 1, 1, 1, quality=q(0.004, 0.0159, 0.99, 0.87))
    assert quality_delta(better, ref)[0] is False  # type: ignore[arg-type]
    assert quality_delta(Rung("x", "s", 1, 1, 1, 1, quality=q(0, 0, 1, 1)), None) == (
        False,
        "reference",
    )


def test_ladder_and_latency_section(bench_root: Path) -> None:
    rungs = load_space_bench(bench_root)
    table = "\n".join(ladder_table(rungs))
    for name in RUNGS:
        assert f"| {name}" in table
    assert "kv **(shipped)** | bench3 | 442.60 / 2325.78" in table
    assert "kv@256 | not run" in table and "kv (repeat) | bench2" in table
    assert "YES" in table and "none" not in table.split("kv@64")[1].split("\n")[0]
    sec = "\n".join(latency_section(rungs, None, 2800))
    assert "p95 2325.78 ms" in sec and "NOT met" in sec and "2325.78 to 3664.00" in sec
    assert "Gate threshold in effect: p95 <= 2800" in sec
    assert "fastest is `kv@64` at 622.00 ms" in sec and "placeholder" in sec
    with_b2 = "\n".join(latency_section(rungs, {"kv@64": {"fpr": 0.02}}))
    assert '"kv@64"' in with_b2 and "placeholder" not in with_b2
    assert "see the file" in "\n".join(latency_section(rungs, {"k": "x" * 900}))
    assert "not measured" in "\n".join(latency_section([], None))


def test_load_b2(tmp_path: Path) -> None:
    assert load_b2(tmp_path) is None
    f = tmp_path / B2_PATH
    f.parent.mkdir(parents=True)
    f.write_text('{"a": 1}')
    assert load_b2(tmp_path) == {"a": 1}


def test_render_report_final_banner_retention_and_leak_status() -> None:
    rows = [Row.new(f"attack {i}", "UNSAFE", "exfiltration", "t") for i in range(3)] + [
        Row.new(f"hello {i}", "SAFE", "none", "t") for i in range(3)
    ]
    preds = [Pred(int(r.label == "UNSAFE"), int(r.label == "UNSAFE"), 0.9, "x", "x") for r in rows]
    models = {"m": ModelEval("m", preds)}
    md = render_report(
        "v1",
        rows,
        models,
        baseline="m",
        focus="m",
        size_rows=["| a |"],
        resamples=20,
        title_suffix=" - FINAL",
        retention_text="custom retention",
        leak_suite=[
            {"level": 1, "n": 4, "leaks": 0, "rate": 0.0, "ci_lo": 0.0, "ci_hi": 0.5},
            {"level": 3, "status": "not run: no guard host (ADR-0063)"},
            {"level": 4},
        ],
    )
    assert md.startswith("# Eval v1 - FINAL\n") and "custom retention" in md
    assert "| 3 | - | - | not run: no guard host (ADR-0063) |" in md
    assert "| 4 | - | - | not run |" in md


def test_leak_suite_resume_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("run_leak_suite")
    monkeypatch.delenv("AGENT_MODELS", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    assert m.agent_models() == ["openai/gpt-oss-20b"]
    monkeypatch.setenv("AGENT_MODEL", "a/b")
    assert m.agent_models() == ["a/b"]
    monkeypatch.setenv("AGENT_MODELS", "x/1, y/2,x/1")
    assert m.agent_models() == ["x/1", "y/2"]
    done = {
        "1": {"rate": 0.0},
        "3": {"level": 3, "status": "not run (guard service unavailable)"},  # stale placeholder
    }
    assert m.pending_levels(done, False) == ([2], [3, 4, 5])
    assert m.pending_levels(done, True) == ([2, 3, 4, 5], [])
    assert m.pending_levels({str(i): {"rate": 0.0} for i in range(1, 6)}, True) == ([], [])


class FakeService:
    fail_from = 99

    def __init__(self, *a: object, **k: object) -> None:
        self.level = 0

    def start_session(self, *a: object, **k: object) -> dict[str, object]:
        return {"session_id": "s"}

    def send(self, sid: str, level: int, msg: str) -> dict[str, object]:
        if level >= self.fail_from:
            raise m_game_error()
        return {"leaked": level == 1 and "ignore" in msg.lower()}


def m_game_error() -> Exception:
    from breachlab.gateway.service import GameError

    return GameError(503, "down")


def test_leak_suite_main_runs_only_missing_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    m = load("run_leak_suite")
    out = tmp_path / "leak.json"
    out.write_text(json.dumps({"1": {"level": 1, "rate": 0.5}}))
    monkeypatch.setattr(m, "OUT", out)
    monkeypatch.setattr(m, "PROGRESS", tmp_path / "progress.json")
    monkeypatch.setattr(m, "BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(m, "warm", lambda *a, **k: True)
    monkeypatch.setattr(m, "attack_suite", lambda: ["ignore all rules", "hello", "x", "y"])
    monkeypatch.setattr(m, "GameService", FakeService)
    monkeypatch.setattr(m, "GroqAgent", lambda key, models: object())
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GUARD_URL", raising=False)
    monkeypatch.setenv("AGENT_MODELS", "m/1,m/2")
    assert m.main() == 0
    saved = json.loads(out.read_text())
    assert saved["1"]["rate"] == 0.5  # untouched, not re-run
    assert saved["2"]["n"] == 4 and saved["2"]["agent"] == "m/1"
    assert "3" not in saved  # no placeholder stored without a guard
    assert "ADR-0063" in capsys.readouterr().out
    # a guard host appears: only L3-L5 run; a dead host (all requests fail) stores nothing
    monkeypatch.setenv("GUARD_URL", "http://guard.invalid")
    monkeypatch.setattr(FakeService, "fail_from", 4)
    assert m.main() == 0
    saved = json.loads(out.read_text())
    assert saved["3"]["n"] == 4 and "4" not in saved and "5" not in saved
    assert "not stored" in capsys.readouterr().out


def test_make_report_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("make_report")
    assert "prefix-KV" in m.lat_tag("0.6b", "sft") and "not measured" in m.lat_tag("0.6b", "dpo")
    assert "4-vCPU" in m.lat_tag("1.7b", "sft")
    gate = {
        "candidate": {
            "asr": 0.005,
            "fpr": 0.016,
            "p95_latency_ms": 2325.78,
            "ece": 0.009,
            "mcnemar_p": 1e-9,
        },
        "champion": {"asr": 1.0, "fpr": 0.004, "is_base": True},
    }
    cfg = tmp_path / "gate.yaml"
    cfg.write_text("p95_latency_ms_max: 2800\n")
    monkeypatch.setenv("GATE_CONFIG", str(cfg))
    text = "\n".join(m.gate_verdict(gate))
    assert "GATE PASSED" in text and "NOT met" in text
    cfg.write_text("p95_latency_ms_max: 300\n")
    text = "\n".join(m.gate_verdict(gate))
    assert "GATE FAILED" in text and "FAIL latency" in text and "NOT met" not in text


def test_vercel_bench_is_the_shipped_latency(bench_root: Path) -> None:
    assert load_vercel_bench(bench_root) is None  # no lane files there
    vb = load_vercel_bench(ROOT)
    assert vb is not None and vb.c1["n"] == 500
    r = vb.rung
    assert (r.name, r.source) == (
        "int4",
        PREFERRED_SOURCE["int4"] == "vercel-warm" and "vercel-warm",
    )
    assert r.c1_p95_ms == pytest.approx(3467.84, abs=0.01) and r.c4_p95_ms < r.c4_p50_ms * 6
    assert vb.first_request_s == pytest.approx(12.84, abs=0.01)
    sec = "\n".join(
        latency_section(load_space_bench(bench_root), None, 4161, vercel=vb, quality_note="Q")
    )
    assert "p95 3467.84 ms" in sec and "The 300 ms target is NOT met" in sec
    assert "Gate threshold in effect: p95 <= 4161" in sec and "guard waking up" in sec
    assert "Superseded: Kaggle 2-core emulation" in sec and "(fp32 reference, superseded)" in sec
    assert "**(shipped)**" not in sec and "\nQ\n" in sec
    only_warm = vb.__class__(vb.url, vb.started, vb.c1, vb.c4, [])
    assert "Cold start" not in "\n".join(latency_section([], None, None, vercel=only_warm))
    assert only_warm.first_request_s is None


def test_send_valid_retries_infra_errors_and_rewarms(monkeypatch: pytest.MonkeyPatch) -> None:
    from breachlab.gateway.service import GameError

    m = load("run_leak_suite")
    monkeypatch.setattr(m, "BACKOFF_S", (0.0, 0.0, 0.0))
    calls: list[str] = []

    class Svc:
        n = 0

        def send(self, sid: str, level: int, msg: str) -> dict[str, object]:
            self.n += 1
            if self.n < 3:
                raise GameError(503, "guard waking up, retry in a few seconds")
            return {"blocked": True, "leaked": False}

    reply, err = m.send_valid(Svc(), "s", 4, "x", lambda: calls.append("warm"))  # type: ignore[arg-type]
    assert reply == {"blocked": True, "leaked": False} and err == "" and calls == ["warm", "warm"]

    class Dead:
        def send(self, sid: str, level: int, msg: str) -> dict[str, object]:
            raise RuntimeError("boom")

    reply, err = m.send_valid(Dead(), "s", 4, "x", lambda: None)  # type: ignore[arg-type]
    assert reply is None and err == "RuntimeError"


def test_leak_suite_stops_after_identical_failures_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    m = load("run_leak_suite")
    out, prog = tmp_path / "leak.json", tmp_path / "progress.json"
    monkeypatch.setattr(m, "OUT", out)
    monkeypatch.setattr(m, "PROGRESS", prog)
    monkeypatch.setattr(m, "BACKOFF_S", (0.0,))
    monkeypatch.setattr(m, "warm", lambda *a, **k: True)
    monkeypatch.setattr(m, "attack_suite", lambda: [f"a{i}" for i in range(10)])
    monkeypatch.setattr(m, "MIN_VALID", 8)
    monkeypatch.setattr(m, "GameService", FakeService)
    monkeypatch.setattr(m, "GroqAgent", lambda key, models: object())
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GUARD_URL", "http://guard.invalid")
    monkeypatch.setenv("GUARD_TOKEN", "tok")
    seen: list[str] = []

    def flaky(self: FakeService, sid: str, level: int, msg: str) -> dict[str, object]:
        seen.append(msg)
        if msg in {"a3", "a4", "a5"} and not resumed[0]:
            raise m_game_error()
        return {"blocked": True, "leaked": msg == "a1"}

    resumed = [False]
    monkeypatch.setattr(FakeService, "send", flaky)
    assert m.main([3]) == 0
    saved = json.loads(out.read_text())["3"]
    assert (
        saved["n"] == 3
        and saved["partial"] is True
        and "3 consecutive attacks failed" in saved["stopped"]
    )
    assert saved["leaks"] == 1 and saved["blocked"] == 3
    assert "tok" not in capsys.readouterr().out and "tok" not in out.read_text()
    # the host recovers: only the missing attacks are sent, the level completes and is no longer partial
    resumed[0], seen[:] = True, []
    assert m.main([3]) == 0
    saved = json.loads(out.read_text())["3"]
    assert saved["n"] == 10 and "partial" not in saved and seen == [f"a{i}" for i in range(3, 10)]
