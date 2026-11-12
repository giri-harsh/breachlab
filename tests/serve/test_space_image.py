"""The staged build context really is self-contained and the Dockerfile is well formed (no docker daemon here)."""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

from breachlab.serve import space_build


def test_stage_only_cli_writes_context(tmp_path: Path, capsys: object) -> None:
    dest = tmp_path / "ctx"
    assert space_build.main(["--stage-only", str(dest)]) == 0
    assert (dest / "Dockerfile").exists() and (dest / "breachlab/serve/space_app.py").exists()
    assert (dest / "configs/serve.yaml").exists() and (dest / "app.py").exists()


def test_staged_context_imports_without_the_repo(tmp_path: Path) -> None:
    """Import ``app`` from the staged folder only (breachlab resolved from it) and hit /health, /score."""
    dest = tmp_path / "ctx"
    space_build.stage(dest)
    probe = textwrap.dedent(
        """
        import pathlib, sys
        import breachlab
        here = pathlib.Path(".").resolve()
        assert pathlib.Path(breachlab.__file__).resolve().is_relative_to(here), breachlab.__file__
        from fastapi.testclient import TestClient
        import app as guard
        from breachlab.serve.space_app import Settings, create_app
        empty = pathlib.Path("empty"); empty.mkdir()
        with TestClient(create_app(Settings(model_dir=empty), background=False)) as c:
            assert c.get("/health").status_code == 503      # no weights: fails closed, never open
            assert c.post("/score", json={"text": "hi"}).status_code == 503
        assert guard.app.title == "BreachLab guard"
        print("closure-ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=dest,
        env={"PYTHONPATH": str(dest), "BREACHLAB_CONFIG_DIR": str(dest / "configs"), "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0 and "closure-ok" in out.stdout, out.stderr[-1500:]


def test_dockerfile_syntax() -> None:
    """Instruction-level lint: known instructions only, FROM first, exec-form CMD parses as JSON, port honours $PORT."""
    import json

    text = (space_build.SPACE_SRC / "Dockerfile").read_text()
    logical = re.sub(r"\\\n", " ", text)
    known = {"FROM", "RUN", "ENV", "WORKDIR", "COPY", "USER", "EXPOSE", "CMD", "ARG", "HEALTHCHECK"}
    instrs = [ln.strip() for ln in logical.splitlines() if ln.strip() and not ln.startswith("#")]
    assert instrs[0].startswith("FROM python:3.11")
    for ln in instrs:
        assert ln.split()[0] in known, ln
    cmd = next(ln for ln in instrs if ln.startswith("CMD"))
    argv = json.loads(cmd.removeprefix("CMD").strip())
    assert argv[0] == "sh" and "${PORT:-7860}" in argv[-1] and "--workers 1" in argv[-1]
    copies = [ln for ln in instrs if ln.startswith("COPY")]
    assert any("requirements.txt" in c for c in copies)  # dependency layer is cached separately
    assert instrs.index(copies[0]) < instrs.index(
        next(i for i in instrs if i.startswith("RUN pip"))
    )


def test_requirements_cover_runtime_imports() -> None:
    reqs = (space_build.SPACE_SRC / "requirements.txt").read_text().lower()
    for pkg in ("fastapi", "uvicorn", "pydantic", "pydantic-settings", "numpy", "onnxruntime"):
        assert pkg in reqs
    assert "tokenizers" in reqs and "huggingface_hub" in reqs


def test_parse_secrets_keeps_deploy_and_runtime_tokens_apart() -> None:
    got = space_build.parse_secrets(["HF_TOKEN=HF_READ_TOKEN", "GUARD_TOKEN"])
    assert got == {"HF_TOKEN": "HF_READ_TOKEN", "GUARD_TOKEN": "GUARD_TOKEN"}
    assert space_build.parse_secrets([]) == {}
