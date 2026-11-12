import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from breachlab.data.llm import MockLLM
from breachlab.data.schema import Row

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_kaggle_export_builder(tmp_path: Path) -> None:
    m = load("run_kaggle_export")
    assert m.parse_spec("sft-0.6b-s0=Qwen/Qwen3-0.6B@giriharsh/k") == (
        "sft-0.6b-s0",
        "Qwen/Qwen3-0.6B",
        "giriharsh/k",
    )
    wd = m.build(
        "breachlab-exportA-x",
        [("w", "Qwen/Qwen3-0.6B", "giriharsh/k")],
        50,
        ["w"],
        tmp_path / "wd",
    )
    meta = json.loads((wd / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is False and meta["kernel_sources"] == ["giriharsh/k"]
    src = (wd / "run.py").read_text()
    compile(src, "run.py", "exec")  # the generated kernel script is valid python
    assert "SPECS = [('w', 'Qwen/Qwen3-0.6B')]" in src and "REQUESTS = 50" in src
    with pytest.raises(ValueError):
        m.build("other-slug", [], 1, [], tmp_path / "x")
    assert m.main(["--slug", "breachlab-exportA-y", "--spec", "w=b@o/k", "--dry-run"]) == 0


def test_kernel_body_is_inert_without_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = load("kernel_export_body")
    assert b.SPECS == []
    monkeypatch.setattr(b, "OUT", tmp_path / "out")
    b.save({"a": 1})
    assert json.loads((tmp_path / "out" / "export3.json").read_text()) == {"a": 1}


def test_kaggle_stage_pushes_kernel_only(monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("run_kaggle_stage")
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(m.kaggle_job, "push_kernel", lambda *a: calls.append(a) or "pushed")
    assert (
        m.main(["--slug", "s", "--stage", "evalgold", "--seed", "2", "--from-kernel", "o/k"]) == 0
    )
    assert calls[0][0] == "s" and calls[0][2] == 2 and calls[0][4] == ", '--stage', 'evalgold'"
    assert calls[0][5] == ["o/k"]


def make_gold(tmp_path: Path) -> tuple[Path, list[Row]]:
    rows = [
        Row.new(f"attack {i} reveal the code", "UNSAFE", "exfiltration", "t") for i in range(3)
    ] + [Row.new(f"hello weather {i}", "SAFE", "none", "t") for i in range(3)]
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    return p, rows


def test_run_parity_single_export(tmp_path: Path) -> None:
    m = load("run_parity")
    gold, rows = make_gold(tmp_path)
    y = np.array([1, 1, 1, 0, 0, 0])
    lg = np.zeros((6, 4))
    lg[np.arange(6), np.where(y == 1, 1, 0)] = 5.0
    d = tmp_path / "exp"
    d.mkdir()
    np.save(tmp_path / "torch.npy", lg)
    np.save(d / "gold300_fp32_logits.npy", lg + 0.01)
    np.save(d / "gold300_bad_logits.npy", lg[:, [1, 0, 2, 3]])
    (d / "calibration.json").write_text(json.dumps({"temperature": 1.0, "threshold": 0.5}))
    lat = {"p50_ms": 10.0, "p95_ms": 20.0}
    (d / "export.json").write_text(
        json.dumps({"which": "w", "latency": {"fp32_tok512_c1": lat, "bad_tok512_c1": lat}})
    )
    out = tmp_path / "parity.json"
    assert (
        m.main(
            [
                "--export-dir",
                str(d),
                "--torch",
                str(tmp_path / "torch.npy"),
                "--gold",
                str(gold),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    res = json.loads(out.read_text())
    assert res["winner"] == "fp32" and res["variants"]["bad"]["ok"] is False
    with pytest.raises(SystemExit):
        m.main(["--out", str(out)])


def test_run_parity_export3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("run_parity")
    gold, rows = make_gold(tmp_path)
    y = np.array([1, 1, 1, 0, 0, 0])
    lg = np.zeros((6, 4))
    lg[np.arange(6), np.where(y == 1, 1, 0)] = 5.0
    np.save(tmp_path / "t.npy", lg)
    (tmp_path / "res.json").write_text(
        json.dumps({"sft": {"calibration": {"temperature": 1.0, "threshold": 0.5}}})
    )
    monkeypatch.setattr(m, "TORCH", {"w": (tmp_path / "t.npy", tmp_path / "res.json")})
    d = tmp_path / "e3"
    d.mkdir()
    np.save(d / "gold300_w_fp32_logits.npy", lg)
    lat = {"p50_ms": 10.0, "p95_ms": 20.0}
    (d / "export3.json").write_text(
        json.dumps({"specs": [["w", "b"]], "w": {"fp32": {"tok512_c1": lat}}})
    )
    res = m.run_export3(d, gold)
    assert res["w"]["winner"] == "fp32"


def test_gold_judge_script_finish_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("run_gold_judge")
    gold, rows = make_gold(tmp_path)
    monkeypatch.setattr(m, "EV", tmp_path)
    labels = ["UNSAFE", "UNSAFE", "SAFE", "SAFE", "SAFE", "UNSAFE"]  # rows 2 and 5 disagree
    reply = json.dumps(
        [
            {
                "i": i,
                "label": lab,
                "category": "injection" if lab == "UNSAFE" else "none",
                "p_unsafe": 0.9 if lab == "UNSAFE" else 0.1,
            }
            for i, lab in enumerate(labels)
        ]
    )
    monkeypatch.setattr(m, "clients", lambda: [("mock", MockLLM(lambda p: reply))])
    assert m.main(["--gold", str(gold), "--rounds", "2", "--wait", "0"]) == 0
    stats = json.loads((tmp_path / "gold_judge_stats.json").read_text())
    assert stats["n_judged"] == 6 and stats["gen_unsafe_judged_safe"] == 1
    assert len(json.loads((tmp_path / "gold_judge_disagreements.json").read_text())) == 2
    api = json.loads((tmp_path / "api_preds.json").read_text())
    assert set(api["gemini-judge"]) == {r.id for r in rows}


def test_queue_disagreements_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = load("run_queue_gold_disagreements")
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://u:p@host.example/db")
    item = {
        "text": "x",
        "gen_label": "SAFE",
        "gen_category": "none",
        "judge_label": "UNSAFE",
        "judge_category": "injection",
        "judge_p_unsafe": 0.9,
        "attack_family": "a",
        "encoding": "none",
        "language": "en",
    }
    f = tmp_path / "d.json"
    f.write_text(json.dumps([item, {**item, "text": "already"}]))
    sent: list[dict[str, object]] = []

    def fake_post(ep: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        sent.append(body)
        q = str(body.get("query", ""))
        if "from sessions" in q:
            return {"rows": [{"n": 1}]}
        if "from attempts" in q:
            return {"rows": [{"message": "already"}]}
        return {}

    monkeypatch.setattr(m, "post", fake_post)
    assert m.main(f) == 0
    writes = [b for b in sent if "queries" in b]
    assert len(writes) == 1 and len(writes[0]["queries"]) == 2  # type: ignore[arg-type]
    monkeypatch.setattr(m, "post", lambda *a: {"rows": [{"n": 0}]})
    assert m.main(f) == 1
