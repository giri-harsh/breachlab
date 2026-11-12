"""Reproducibility tags (PRD 12): git SHA, config hash, data version, seed. No network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from breachlab.common.config import TrainConfig
from breachlab.train import kaggle_job, repro


def cfg(**kw: Any) -> TrainConfig:
    return TrainConfig(**kw)


def test_git_sha_prefers_env(tmp_path: Path) -> None:
    (tmp_path / "GIT_SHA").write_text("filesha\n")
    env = {"BREACHLAB_GIT_SHA": " envsha ", "GITHUB_SHA": "ghsha"}
    assert repro.git_sha(env, tmp_path, lambda _: "revsha") == "envsha"
    assert repro.git_sha({"GITHUB_SHA": "ghsha"}, tmp_path, lambda _: "revsha") == "ghsha"


def test_git_sha_file_then_rev_parse_then_unknown(tmp_path: Path) -> None:
    (tmp_path / "GIT_SHA").write_text("filesha\n")
    assert repro.git_sha({}, tmp_path, lambda _: "revsha") == "filesha"
    (tmp_path / "GIT_SHA").write_text("  \n")
    assert repro.git_sha({}, tmp_path, lambda _: "revsha") == "revsha"
    assert repro.git_sha({}, tmp_path / "nowhere", lambda _: None) == "unknown"


def test_git_rev_parse_safe_outside_a_repo(tmp_path: Path) -> None:
    assert repro._git_rev_parse(tmp_path) is None
    assert repro._git_rev_parse(tmp_path / "missing-dir") is None


def test_config_hash_ignores_location_but_not_settings(tmp_path: Path) -> None:
    a = cfg(data_dir=tmp_path / "a", output_dir=tmp_path / "o1")
    b = cfg(data_dir=Path("/kaggle/input/x"), output_dir=Path("/kaggle/working/out"))
    assert repro.config_sha256(a) == repro.config_sha256(b)
    assert len(repro.config_sha256(a)) == 64
    assert repro.config_sha256(cfg(seed=1)) != repro.config_sha256(cfg(seed=0))
    c = cfg()
    c.sft.precision = "bf16" if c.sft.precision != "bf16" else "fp16"
    assert repro.config_sha256(c) != repro.config_sha256(cfg())
    assert "data_dir" not in repro.config_payload(a)


def test_data_sha256(tmp_path: Path) -> None:
    assert repro.data_sha256(tmp_path) is None
    (tmp_path / "train.jsonl").write_text('{"a": 1}\n')
    h1 = repro.data_sha256(tmp_path)
    (tmp_path / "val.jsonl").write_text('{"b": 2}\n')
    h2 = repro.data_sha256(tmp_path)
    assert h1 and h2 and h1 != h2
    assert repro.data_sha256(tmp_path) == h2


def test_repro_tags_and_logging(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("x\n")
    c = cfg(seed=2, data_dir=tmp_path)

    class Sink:
        got: dict[str, Any] = {}

        def set_tags(self, tags: dict[str, Any]) -> None:
            self.got = dict(tags)

    s = Sink()
    tags = repro.log_repro_tags(c, s, sha="abc123")
    assert s.got == tags
    assert tags["git_sha"] == "abc123"
    assert tags["seed"] == "2"
    assert tags["data_version"] == "data-v1"
    assert tags["config_sha256"] == repro.config_sha256(c)
    assert "data_sha256" in tags
    assert repro.missing_tags(tags) == []
    assert "data_sha256" not in repro.repro_tags(cfg(data_dir=tmp_path / "none"), sha="x")


def test_missing_tags() -> None:
    assert repro.missing_tags({}) == ["git_sha", "config_sha256", "data_version", "seed"]
    assert repro.missing_tags(
        {"git_sha": "unknown", "config_sha256": "h", "data_version": "data-v1", "seed": "0"}
    ) == ["git_sha"]


def test_stage_code_writes_git_sha(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("BREACHLAB_GIT_SHA", "deadbeef")
    kaggle_job.stage_code(tmp_path / "code")
    assert (tmp_path / "code" / "GIT_SHA").read_text().strip() == "deadbeef"
    staged = tmp_path / "code"
    assert repro.git_sha({}, staged, lambda _: None) == "deadbeef"
