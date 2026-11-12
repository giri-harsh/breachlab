"""scripts/check_links.py with tmp files and a fake HEAD (no network)."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_links", ROOT / "scripts" / "check_links.py")
assert spec and spec.loader
cl = importlib.util.module_from_spec(spec)
sys.modules["check_links"] = cl
spec.loader.exec_module(cl)


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "# Title\n\n"
        "[ok doc](docs/a.md) [anchor ok](docs/a.md#second-part) [anchor bad](docs/a.md#nope)\n"
        "[missing](docs/gone.md) [same file](#title) [web](https://good.test/x)\n"
        "[private](https://github.com/giri-harsh/Breach-lab/pull/3)\n"
        "```\ncurl https://in-code-block.test/never [x](nowhere.md)\n```\n"
        "Bare https://bare.test/page. and <https://auto.test/y> and https://localhost:8000/x.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "a.md").write_text(
        "# A\n\n## Second part\n\n[up](../README.md) [bad up](../nope.md#x) "
        "[hf](https://huggingface.co/Pseudo-Harsh/breachlab-guard-v1)\n",
        encoding="utf-8",
    )
    return tmp_path


def test_extract_skips_fenced_code_and_finds_bare_and_auto() -> None:
    text = "[a](x.md) https://one.test/p, <https://two.test> [b](https://three.test)\n```\nhttps://no.test\n```\n"
    targets, urls = cl.extract(text)
    assert targets == ["x.md", "https://three.test"]
    assert "https://one.test/p" in urls and "https://two.test" in urls
    assert not any("no.test" in u for u in urls)


def test_slugify_matches_github_rules() -> None:
    assert cl.slugify("Second part") == "second-part"
    assert cl.slugify("`GUARD_TOKEN` rotation (v2)!") == "guard_token-rotation-v2"
    assert cl.slugify("Hosting: the *guard*") == "hosting-the-guard"


def test_relative_links_and_anchors_offline(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    rep = cl.run(root, head=lambda u, t: pytest.fail("no network expected"), offline=True)
    broken = "\n".join(rep.broken)
    assert "docs/gone.md" in broken and "missing anchor #nope" in broken
    assert "../nope.md#x" in broken
    assert "nowhere.md" not in broken  # inside a fenced block
    assert rep.relative_ok == 4  # a.md, a.md#second-part, #title, ../README.md
    assert not rep.clean
    assert rep.failed == []
    assert any("good.test" in u for u in rep.unchecked)
    assert any("github.com/giri-harsh" in u and "private, not checked" in u for u in rep.private)
    assert any("huggingface.co/Pseudo-Harsh" in u for u in rep.private)
    assert any("localhost" in u for u in rep.skipped)


def test_external_status_handling(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    calls: list[str] = []
    codes = {"https://good.test/x": 200, "https://bare.test/page": 404, "https://auto.test/y": 405}

    def head(url: str, timeout: float) -> int:
        calls.append(url)
        assert timeout == 10.0
        return codes[url]

    rep = cl.run(root, head=head)
    assert [(u, s) for u, s, _ in rep.failed] == [("https://bare.test/page", "404")]
    assert rep.ok_urls == 2  # 200 and 405 (host reachable, HEAD refused)
    assert all("github.com" not in c and "huggingface" not in c for c in calls)
    assert not any("in-code-block" in c for c in calls)


def test_retry_once_on_transport_error_and_5xx() -> None:
    seen: list[int] = []

    def flaky(url: str, timeout: float) -> int:
        seen.append(1)
        if len(seen) == 1:
            raise OSError("boom")
        return 200

    assert cl.check_external("https://x.test", flaky) == (True, "200")
    assert len(seen) == 2

    always = [0]

    def down(url: str, timeout: float) -> int:
        always[0] += 1
        return 503

    ok, status = cl.check_external("https://x.test", down)
    assert (ok, status) == (False, "503") and always[0] == 2

    notfound = [0]

    def gone(url: str, timeout: float) -> int:
        notfound[0] += 1
        return 404

    assert cl.check_external("https://x.test", gone) == (False, "404") and notfound[0] == 1

    def dead(url: str, timeout: float) -> int:
        raise OSError("refused")

    ok, status = cl.check_external("https://x.test", dead)
    assert not ok and "refused" in status


def test_bare_api_host_is_probed_through_its_health_route() -> None:
    assert (
        cl.probe_url("https://breachlab-api.vercel.app")
        == "https://breachlab-api.vercel.app/api/health"
    )
    assert (
        cl.probe_url("https://breachlab-guard.vercel.app/")
        == "https://breachlab-guard.vercel.app/health"
    )
    assert (
        cl.probe_url("https://breachlab-guard.vercel.app/info")
        == "https://breachlab-guard.vercel.app/info"
    )
    assert cl.probe_url("https://other.test") == "https://other.test"


def test_classify_private_kaggle_and_example_hosts() -> None:
    assert cl.classify("https://www.kaggle.com/code/giriharsh/breachlab-zs-06b") == "private"
    assert cl.classify("https://pseudo-harsh-breachlab-guard.hf.space") == "private"
    assert cl.classify("https://203-0-113-7.sslip.io") == "skip"
    assert cl.classify("https://placeholder.invalid") == "skip"
    assert cl.classify("https://host.test/<x>") == "skip"
    assert cl.classify("https://breachlab-web.vercel.app") == "check"


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_repo(tmp_path)
    assert cl.main(["--root", str(root), "--offline"]) == 1
    out = capsys.readouterr().out
    assert "BROKEN" in out and "PRIVATE" in out
    (root / "README.md").write_text("# T\n\n[a](docs/a.md)\n", encoding="utf-8")
    (root / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    assert cl.main(["--root", str(root), "--offline"]) == 0
