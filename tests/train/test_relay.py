import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from breachlab.train import export_job, export_onnx, relay


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def upload_file(self, *, path_or_fileobj: str, path_in_repo: str, repo_id: str) -> None:
        # the file must exist at upload time (it is deleted afterwards)
        self.calls.append((path_in_repo, repo_id, Path(path_or_fileobj).exists()))


def make_runner(files: dict[str, bytes], seen: list[Sequence[str]]) -> relay.Runner:
    def run(args: Sequence[str]) -> None:
        seen.append(args)
        root = Path(args[args.index("-p") + 1])
        pat = re.compile(args[args.index("--file-pattern") + 1])
        for name, data in files.items():
            if pat.search(name):
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        (root / "k.log").write_text("log")  # the CLI always drops the kernel log

    return run


def test_repo_path_mapping() -> None:
    job = relay.RelayJob("o/k", ".*", "out/export-x", "champion")
    assert relay.repo_path(job, "out/export-x/guard-fp32.onnx") == "champion/guard-fp32.onnx"
    assert relay.repo_path(job, "out\\export-x\\a.json") == "champion/a.json"
    with pytest.raises(ValueError):
        relay.repo_path(job, "elsewhere/a.json")
    flat = relay.RelayJob("o/k", ".*", "", "")
    assert relay.repo_path(flat, "a.json") == "a.json"


def test_relay_uploads_matching_files_and_cleans_up(tmp_path: Path) -> None:
    files = {
        "out/export-sft-0.6b-s0/guard-fp32.onnx": b"a",
        "out/export-sft-0.6b-s0/guard-fp32.onnx.data": b"bb",
        "out/export-sft-0.6b-s0/calibration.json": b"{}",
        "out/export-sft-0.6b-s0/gold300_fp32_logits.npy": b"no",
        "out/export-dpo-0.6b-s0/guard-fp32.onnx": b"other",
    }
    seen: list[Sequence[str]] = []
    api = FakeApi()
    job = relay.export_job("o/k", "sft-0.6b-s0", "champion")
    sent = relay.relay_job(job, api, "ns/repo", make_runner(files, seen), tmp_path, lambda _: None)
    assert sorted(sent) == [
        "champion/calibration.json",
        "champion/guard-fp32.onnx",
        "champion/guard-fp32.onnx.data",
    ]
    assert all(c[1] == "ns/repo" and c[2] for c in api.calls)
    assert list(tmp_path.iterdir()) == []  # temp dir removed, nothing left behind
    assert "--file-pattern" in seen[0]


def test_relay_fails_when_nothing_matches_and_still_cleans(tmp_path: Path) -> None:
    job = relay.adapter_job("o/k", "sft-0.6b-s0")
    with pytest.raises(FileNotFoundError):
        relay.relay_job(job, FakeApi(), "ns/r", make_runner({}, []), tmp_path, lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_relay_upload_error_cleans_up(tmp_path: Path) -> None:
    class Boom(FakeApi):
        def upload_file(self, **_: Any) -> None:
            raise RuntimeError("429")

    files = {"out/sft-0.6b-s0/adapter/adapter_model.safetensors": b"w"}
    job = relay.adapter_job("o/k", "sft-0.6b-s0")
    with pytest.raises(RuntimeError):
        relay.relay_job(job, Boom(), "ns/r", make_runner(files, []), tmp_path, lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_adapter_job_never_matches_checkpoints() -> None:
    job = relay.adapter_job("o/k", "sft-0.6b-s0")
    pat = re.compile(job.pattern)
    assert pat.search("out/sft-0.6b-s0/adapter/adapter_model.safetensors")
    assert not pat.search("out/sft-0.6b-s0/ckpt/checkpoint-606/adapter_model.safetensors")
    assert not pat.search("out/sft-0.6b-s0/adapter/optimizer.pt")
    assert relay.repo_path(job, "out/sft-0.6b-s0/adapter/adapter_config.json") == (
        "adapters/sft-0.6b-s0/adapter_config.json"
    )


def test_bad_regex_fails_before_download(tmp_path: Path) -> None:
    with pytest.raises(re.error):
        relay.fetch(relay.RelayJob("o/k", "(", "", ""), tmp_path, make_runner({}, []))


def test_guard_plan_layout() -> None:
    plan = relay.guard_plan({"champion": "o/custom"})
    assert plan["champion"].kernel == "o/custom" and plan["champion"].dest == "champion"
    assert {plan[a].dest for a in ("champion", "challenger", "base")} == {
        "champion",
        "challenger",
        "base",
    }
    assert plan["adapter-sft-1.7b"].dest == "adapters/sft-1.7b-s0"


def test_calibration_lookup() -> None:
    sft = {"sft": {"calibration": {"temperature": 1.4, "threshold": 0.06, "ece_after": 0.01}}}
    dpo = {**sft, "dpo": {"calibration": {"temperature": 1.2, "threshold": 0.03}}}
    assert export_onnx.calibration_for("sft-0.6b-s0", [dpo])["temperature"] == 1.4
    assert export_onnx.calibration_for("dpo-0.6b-s0", [sft, dpo])["threshold"] == 0.03
    assert export_onnx.calibration_for("base", []) == export_onnx.BASE_CALIBRATION
    with pytest.raises(LookupError):
        export_onnx.calibration_for("dpo-0.6b-s0", [sft])


def test_load_results_and_manifest(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "results.json").write_text(json.dumps({"sft": 1}))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "results.json").write_text("not json")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "results.json").write_text("[1]")
    assert export_onnx.load_results(tmp_path) == [{"sft": 1}]
    (tmp_path / "x.bin").write_bytes(b"123")
    assert export_onnx.manifest(tmp_path) == {"x.bin": 3}


def test_head_maths() -> None:
    h = np.arange(6, dtype=np.float64).reshape(2, 3)
    head = np.ones((4, 3))
    assert export_onnx.score_from_hidden(h, head).shape == (2, 4)


def test_export_script_and_kernel_selection(tmp_path: Path) -> None:
    script = export_job.export_script("base", "Qwen/Qwen3-0.6B", ("fp32", "w8"), -1)
    compile(script, "run.py", "exec")  # valid python
    assert "which='base'" in script and "parity_rows=-1" in script
    assert export_job.kernels_for("base") == []
    assert export_job.kernels_for("dpo-0.6b-s0") == ["giriharsh/breachlab-dpo-06b-s0"]
    export_job.stage_export_code(tmp_path / "code")
    meta = json.loads((tmp_path / "code" / "dataset-metadata.json").read_text())
    assert meta["id"] == "giriharsh/breachlab-export-code" and meta["isPrivate"] is True
    assert (tmp_path / "code" / "src" / "breachlab").is_dir()
