from pathlib import Path

from breachlab.data.schema import Row
from breachlab.serve import bench_runner


def test_merge_dirs_first_source_wins(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "x.onnx").write_text("A")
    (b / "x.onnx").write_text("B")
    (b / "y.npz").write_text("Y")
    (b / "sub").mkdir()
    linked = bench_runner.merge_dirs([a, b], tmp_path / "out")
    assert sorted(linked) == ["x.onnx", "y.npz"]
    assert (tmp_path / "out" / "x.onnx").read_text() == "A"
    assert (tmp_path / "out" / "y.npz").is_symlink()


def test_collect_models_maps_export_dirs_to_aliases(tmp_path: Path) -> None:
    for kernel, d, f in [
        ("k1", "export-sft-0.6b-s0", "guard-fp32.onnx"),
        ("k2", "export-sft-0.6b-s0", "guard-kv.onnx"),
        ("k3", "export-base", "guard-fp32.onnx"),
    ]:
        p = tmp_path / "in" / kernel / "out" / d
        p.mkdir(parents=True)
        (p / f).write_text("w")
    out = bench_runner.collect_models(tmp_path / "in", tmp_path / "models")
    assert sorted(out["champion"]) == ["guard-fp32.onnx", "guard-kv.onnx"]
    assert "challenger" not in out and out["base"] == ["guard-fp32.onnx"]


def test_bench_job_script_is_valid_python() -> None:
    from breachlab.serve import bench_job

    script = bench_job.bench_script(["--variants", "fp32,kv", "--threads", "2"])
    compile(script, "run.py", "exec")
    assert "onnxruntime==1.31.1" in script and "'--variants', 'fp32,kv'" in script
    assert all(s.startswith("giriharsh/breachlab-export3-") for s in bench_job.DEFAULT_SOURCES)


def test_split_spec() -> None:
    assert bench_runner.split_spec("kv") == ("kv", None)
    assert bench_runner.split_spec("kv@128") == ("kv", 128)


def test_quality_from_service_outputs() -> None:
    rows = [Row.new("a", "SAFE", "none", "t"), Row.new("b", "UNSAFE", "injection", "t")]
    outs = [
        {"label": "SAFE", "category": "none", "p_unsafe": 0.1},
        {"label": "UNSAFE", "category": "injection", "p_unsafe": 0.9},
    ]
    q = bench_runner.quality(rows, outs)
    assert q["auroc"] == 1.0 and q["fpr"] == 0.0 and q["asr"] == 0.0
