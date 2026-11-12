import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "readme_numbers", ROOT / "scripts" / "readme_numbers.py"
)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CARD = {
    "models": {
        "m-1": {"asr": 0.0044, "fpr": 0.017, "f1": 0.98812, "auroc": 0.9991, "p95_ms": 212.4}
    }
}


def test_fill_and_check() -> None:
    text = "ASR <!--num:m-1.asr-->9.9%<!--/num--> and F1 <!--num:m-1.f1-->0.988<!--/num--> p95 <!--num:m-1.p95_ms-->x<!--/num-->"
    new, problems = mod.process(text, CARD, write=True)
    assert "<!--num:m-1.asr-->0.4%<!--/num-->" in new and "212 ms" in new and len(problems) == 2
    again, problems2 = mod.process(new, CARD, write=False)
    assert again == new and problems2 == []
    assert mod.fmt("p95_ms", None) == "n/a" and mod.fmt("auroc", 0.99912) == "0.999"
