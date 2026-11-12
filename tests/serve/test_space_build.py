import ast
from pathlib import Path

from breachlab.serve import space_build


def test_stage_vendors_byte_identical_modules(tmp_path: Path) -> None:
    written = space_build.stage(tmp_path / "space")
    assert set(written) == set(
        space_build.OWN_FILES + space_build.VENDOR + [f"configs/{c}" for c in space_build.CONFIGS]
    )
    for rel in space_build.VENDOR:
        assert (tmp_path / "space" / rel).read_bytes() == (
            space_build.ROOT / "src" / rel
        ).read_bytes()
    readme = (tmp_path / "space" / "README.md").read_text()
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 7860" in readme


def test_stage_is_idempotent(tmp_path: Path) -> None:
    space_build.stage(tmp_path / "s")
    (tmp_path / "s" / "stale.txt").write_text("x")
    space_build.stage(tmp_path / "s")
    assert not (tmp_path / "s" / "stale.txt").exists()


def test_vendor_list_covers_every_runtime_import() -> None:
    """Every breachlab module imported (transitively, at module level) by the Space modules is vendored."""
    src = space_build.ROOT / "src"
    vendored = set(space_build.VENDOR)
    seen: set[str] = set()
    todo = ["breachlab/serve/space_app.py"]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((src / rel).read_text())
        for node in tree.body:  # module level only: lazy imports are inside functions
            names: list[str] = []
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("breachlab")
            ):
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names if a.name.startswith("breachlab")]
            for mod in names:
                path = mod.replace(".", "/")
                cand = f"{path}.py" if (src / f"{path}.py").exists() else f"{path}/__init__.py"
                assert cand in vendored, f"{rel} imports {mod} which is not vendored"
                todo.append(cand)


def test_lazy_runtime_imports_are_vendored() -> None:
    """space_app lazily imports config and the ONNX backend inside functions; those must ship too."""
    for rel in ("breachlab/common/config.py", "breachlab/serve/onnx_backend.py"):
        assert rel in space_build.VENDOR


def test_requirements_pin_onnxruntime() -> None:
    reqs = (space_build.SPACE_SRC / "requirements.txt").read_text()
    assert "onnxruntime==" in reqs
    dockerfile = (space_build.SPACE_SRC / "Dockerfile").read_text()
    assert "7860" in dockerfile and "uvicorn" in dockerfile
