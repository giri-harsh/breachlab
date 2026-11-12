import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_commit_msg_policy() -> None:
    m = _load("check_commit_msg")
    assert m.check("feat: add thing") is None
    assert m.check("feat: thing\n\nCo-Authored-By: x") is not None
    assert m.check("added stuff") is not None
    assert m.check("fix: mention Claude") is not None


def test_check_env_reports_failures() -> None:
    m = _load("check_env")
    res = m.run_checks({"ok": lambda: None, "bad": lambda: "missing env: X"})
    assert res["ok"]["status"] == "PASS" and res["bad"]["status"] == "FAIL"


class _Enc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTok:
    """Single-token iff word in table (bare) or ' '+word in table (space)."""

    def __init__(self, table: dict[str, int]) -> None:
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = False) -> _Enc:
        return _Enc([self.table[text]] if text in self.table else [1, 2])


def test_verbalizer_choose_prefers_bare_then_space() -> None:
    m = _load("verbalizer_check")
    bare = _FakeTok({"safe": 1, "injection": 2, "jailbreak": 3, "exfiltration": 4})
    assert {k: v["form"] for k, v in m.choose(bare).items()} == dict.fromkeys(m.FALLBACKS, "bare")
    spaced = _FakeTok({" safe": 1, " injection": 2, " jailbreak": 3, " exfiltration": 4})
    out = m.choose(spaced)
    assert out["injection"]["word"] == " injection" and out["safe"]["form"] == "space"
    import pytest

    with pytest.raises(RuntimeError):
        m.choose(_FakeTok({"safe": 1}))


def test_ai_trailer_rule() -> None:
    m = _load("check_no_ai_trailers")
    assert m.bad_lines("feat: x\n\nCo-authored-by: Someone <a@b.c>") != []
    assert m.bad_lines("feat: x\n\nClaude-Session: https://claude.ai/code/s") != []
    assert m.bad_lines("feat: add thing\n\nplain body") == []
