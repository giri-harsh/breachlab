import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("link_check", ROOT / "scripts" / "link_check.py")
assert spec and spec.loader
lc = importlib.util.module_from_spec(spec)
sys.modules["link_check"] = lc
spec.loader.exec_module(lc)

LINKS_TS = """\
export const HF_NAMESPACE = process.env.NEXT_PUBLIC_HF_NAMESPACE ?? "Pseudo-Harsh";
export const SPACE_REPO = process.env.NEXT_PUBLIC_HF_SPACE ?? "";
export const HF_MODEL_URL = `https://huggingface.co/${HF_NAMESPACE}/guard`;
export const HF_SPACE_URL = SPACE_REPO ? `https://huggingface.co/spaces/${HF_NAMESPACE}/${SPACE_REPO}` : null;
"""


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "web" / "lib").mkdir(parents=True)
    (tmp_path / "web" / "app").mkdir()
    (tmp_path / "README.md").write_text(
        "See https://example.com/skip and [site](https://breachlab-web.vercel.app).\nRepo: https://github.com/a/b.\n"
    )
    (tmp_path / "web" / "lib" / "links.ts").write_text(LINKS_TS)
    (tmp_path / "web" / "app" / "page.tsx").write_text(
        '<a href="https://huggingface.co/datasets/x/y">d</a> <a href="http://localhost:3000/x">l</a>'
    )
    return tmp_path


def test_readme_links_skip_local_and_trailing_punctuation(tmp_path: Path) -> None:
    got = lc.readme_links(make_repo(tmp_path))
    assert got == ["https://breachlab-web.vercel.app", "https://github.com/a/b"]


def test_web_defaults_and_resolve(tmp_path: Path) -> None:
    d = lc.web_defaults(make_repo(tmp_path))
    assert d == {"HF_NAMESPACE": "Pseudo-Harsh", "SPACE_REPO": ""}
    assert lc.resolve("https://h/${HF_NAMESPACE}/g", d) == "https://h/Pseudo-Harsh/g"
    assert lc.resolve("https://h/${SPACE_REPO}", d) is None  # empty default: skipped
    assert lc.resolve("https://h/${UNKNOWN}", d) is None


def test_web_links_expand_templates_and_drop_unresolved(tmp_path: Path) -> None:
    got = lc.web_links(make_repo(tmp_path))
    assert "https://huggingface.co/Pseudo-Harsh/guard" in got
    assert "https://huggingface.co/datasets/x/y" in got
    assert not any("spaces" in u or "localhost" in u for u in got)


def test_links_union_sorted(tmp_path: Path) -> None:
    got = lc.links(make_repo(tmp_path))
    assert got == sorted(set(got)) and "https://github.com/a/b" in got


def test_real_repo_links_resolve() -> None:
    got = lc.links()
    assert any(u.startswith("https://huggingface.co/Pseudo-Harsh/") for u in got)
    assert all("${" not in u for u in got)


@pytest.mark.parametrize(
    ("url", "status", "strict", "verdict"),
    [
        ("https://x.org", 200, False, "OK"),
        ("https://github.com/a/b", 404, False, "WARN"),
        ("https://github.com/a/b", 404, True, "FAIL"),
        ("https://breachlab-web.vercel.app", 404, False, "FAIL"),
        ("https://huggingface.co/a", None, False, "FAIL"),
    ],
)
def test_classify(url: str, status: int | None, strict: bool, verdict: str) -> None:
    assert lc.classify(url, status, strict) == verdict


def test_main_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        lc,
        "links",
        lambda root=None: ["https://ok.test", "https://github.com/a/b", "https://bad.test"],
    )

    def fake_get(url: str, **_: object) -> httpx.Response:
        if "bad" in url:
            raise httpx.ConnectError("boom")
        return httpx.Response(200 if "ok" in url else 404)

    monkeypatch.setattr(lc.httpx, "get", fake_get)
    assert lc.main() == 1
    out = capsys.readouterr().out
    assert "OK" in out and "WARN" in out and "FAIL" in out
    monkeypatch.setattr(
        lc, "links", lambda root=None: ["https://ok.test", "https://github.com/a/b"]
    )
    assert lc.main() == 0


def test_probe_url_uses_health_for_gateway_root() -> None:
    assert (
        lc.probe_url("https://breachlab-api.vercel.app")
        == "https://breachlab-api.vercel.app/api/health"
    )
    assert (
        lc.probe_url("https://breachlab-api.vercel.app/")
        == "https://breachlab-api.vercel.app/api/health"
    )
    assert lc.probe_url("https://breachlab-web.vercel.app") == "https://breachlab-web.vercel.app"
