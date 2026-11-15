import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from breachlab.eval.metrics import asr, auroc, fpr
from breachlab.train.scoring import probs_from_logits

ROOT = Path(__file__).resolve().parents[2]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_rows(n: int = 40) -> list[dict[str, str]]:
    return [
        {
            "id": f"r{i}",
            "label": "UNSAFE" if i % 2 == 0 else "SAFE",
            "language": "en" if i < n // 2 else "hi",
        }
        for i in range(n)
    ]


def test_parse_model() -> None:
    m = load("ood_report")
    s = m.parse_model("v2=out/x.npy,T=1.5,thr=0.4")
    assert (s.name, s.path, s.temperature, s.threshold) == ("v2", Path("out/x.npy"), 1.5, 0.4)
    d = m.parse_model("c=c.npy")
    assert (d.temperature, d.threshold) == (1.0, 0.5)
    with pytest.raises(ValueError):
        m.parse_model("nopath")
    with pytest.raises(ValueError):
        m.parse_model("a=b.npy,foo=1")


def test_p_unsafe_matches_scoring() -> None:
    m = load("ood_report")
    lg = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    p = m.p_unsafe_from_array(lg, 2.0)
    assert np.allclose(p, 1 - probs_from_logits(lg, 2.0)[:, 0])
    assert np.allclose(
        m.p_unsafe_from_array(np.array([0.1, 0.9]), 5.0), [0.1, 0.9]
    )  # (n,) used as is
    with pytest.raises(ValueError):
        m.p_unsafe_from_array(np.zeros((3, 2)), 1.0)


def test_report_matches_metric_helpers(tmp_path: Path) -> None:
    m = load("ood_report")
    rows = make_rows()
    y = [int(r["label"] == "UNSAFE") for r in rows]
    # perfect scorer except two attacks scored low and one benign scored high
    scores = np.array([0.9 if t else 0.1 for t in y])
    scores[0], scores[2], scores[1] = 0.2, 0.3, 0.8
    f = tmp_path / "s.npy"
    np.save(f, scores)
    res = m.report_model(m.parse_model(f"x={f},thr=0.5"), rows, None, resamples=50)
    preds = m.to_preds(y, scores, 0.5)
    assert res["overall"]["asr"]["value"] == pytest.approx(asr(preds))
    assert res["overall"]["fpr"]["value"] == pytest.approx(fpr(preds))
    assert res["overall"]["auroc"]["value"] == pytest.approx(auroc(preds))
    assert res["overall"]["asr"]["value"] == pytest.approx(2 / 20)
    assert res["overall"]["fpr"]["value"] == pytest.approx(1 / 20)
    for k in ("asr", "fpr", "auroc"):
        v = res["overall"][k]
        assert (
            v["lo"] <= v["value"] <= v["hi"] or k == "auroc"
        )  # percentile CI can exclude a boundary point
    assert set(res["by_language"]) == {"en", "hi"}
    assert res["by_language"]["en"]["n"] == 20


def test_row_count_mismatch_is_rejected() -> None:
    m = load("ood_report")
    with pytest.raises(ValueError, match="file order"):
        m.to_preds([1, 0, 1], np.array([0.5, 0.5]), 0.5)


def test_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = load("ood_report")
    rows = make_rows()
    data = tmp_path / "ood_test.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = tmp_path / "ood_test.meta.jsonl"
    meta.write_text(
        "\n".join(
            json.dumps({"id": r["id"], "origin": "authored" if i % 3 else "public"})
            for i, r in enumerate(rows)
        )
        + "\n"
    )
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(len(rows), 4))
    f = tmp_path / "l.npy"
    np.save(f, logits)
    out = tmp_path / "o.json"
    assert (
        m.main(
            ["--data", str(data), "--model", f"v={f},T=1", "--resamples", "20", "--json", str(out)]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "== v" in text and "language=hi" in text and "origin=public" in text
    saved = json.loads(out.read_text())
    assert saved[0]["overall"]["n"] == 40 and set(saved[0]["by_origin"]) == {"authored", "public"}
