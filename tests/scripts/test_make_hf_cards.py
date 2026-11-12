import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "make_hf_cards", ROOT / "scripts" / "make_hf_cards.py"
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_cards_state_the_honest_facts() -> None:
    f = mod.facts()
    card = mod.model_card(f)
    assert "SFT-only" in card and "prefix-KV" in card
    assert f"T = {f['temperature']}" in card and f["threshold"] in card
    assert "seed 2" in card and "not met" in card.lower() and "in-distribution" in card.lower()
    assert f"{f['p50_kv']} / {f['p95_kv']} ms" in card
    assert "wildjailbreak" in card and "excluded" in card
    assert f"kappa {f['kappa']}" in card
    data = mod.dataset_card(f, (ROOT / "DATASET_CARD.md").read_text(encoding="utf-8"))
    assert "EXCLUDED" in data and "no row comes from it" in data and data.startswith("---\n")
    assert "Live game data is never included" in data


def test_missing_number_stops_the_build(tmp_path: Path) -> None:
    for rel in (
        "web/public/scorecard.json",
        "reports/eval-v1.md",
        "DATASET_CARD.md",
    ):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    assert mod.facts(tmp_path)["n_seeds"] == 3
    report = (tmp_path / "reports/eval-v1.md").read_text(encoding="utf-8")
    (tmp_path / "reports/eval-v1.md").write_text(
        report.replace("concurrency 1: p50", "c1 p50"), encoding="utf-8"
    )
    with pytest.raises(mod.CardError):
        mod.facts(tmp_path)
    shutil.copy(ROOT / "reports/eval-v1.md", tmp_path / "reports/eval-v1.md")
    sc = json.loads((tmp_path / "web/public/scorecard.json").read_text())
    sc["models"]["qwen3-0.6b-sft"]["note"] = ""
    (tmp_path / "web/public/scorecard.json").write_text(json.dumps(sc))
    with pytest.raises(mod.CardError):
        mod.facts(tmp_path)
