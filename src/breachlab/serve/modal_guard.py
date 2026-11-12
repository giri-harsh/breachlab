"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal CPU app serving the champion + challenger ONNX int8 guards (8 vCPU). Not unit-tested.

Deploy: modal deploy src/breachlab/serve/modal_guard.py   (needs persisted secrets: ADR-0022)
Models are pulled from the HF repos named in GUARD_CHAMPION_REPO / GUARD_CHALLENGER_REPO.
"""

import os

import modal

app = modal.App("breachlab-guard")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime>=1.19",
        "tokenizers",
        "numpy",
        "pydantic>=2.7",
        "pydantic-settings",
        "pyyaml",
        "huggingface_hub",
    )
    .env({"BREACHLAB_CONFIG_DIR": "/app/configs"})
    .add_local_python_source("breachlab")
    .add_local_dir("configs", "/app/configs")
)
MIN = int(os.environ.get("GUARD_MIN_CONTAINERS", "0"))


@app.cls(
    image=image,
    cpu=8,
    memory=8192,
    min_containers=MIN,
    secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
    scaledown_window=300,
)
class Guard:
    @modal.enter()
    def load(self) -> None:
        from huggingface_hub import snapshot_download

        from breachlab.common.config import ServeConfig
        from breachlab.serve.guard_runtime import GuardService
        from breachlab.serve.onnx_backend import load_guard

        serve = ServeConfig()
        models = {}
        for alias, env in (
            ("champion", "GUARD_CHAMPION_REPO"),
            ("challenger", "GUARD_CHALLENGER_REPO"),
        ):
            repo = os.environ.get(env)
            if repo:
                path = snapshot_download(repo, token=os.environ.get("HF_TOKEN"))
                models[alias] = load_guard(
                    __import__("pathlib").Path(path), serve, f"{alias}:{repo}"
                )
        self.service = GuardService(models)

    @modal.method()
    def score(self, text: str, kind: str = "input", alias: str = "champion") -> dict[str, object]:
        r = self.service.score(text, "output" if kind == "output" else "input", alias)
        return {
            "label": r.verdict.label,
            "category": r.verdict.category,
            "confidence": r.verdict.p_unsafe,
            "version": r.version,
            "latency_ms": r.latency_ms,
        }

    @modal.method()
    def bench(self, texts: list[str], concurrency: int = 1) -> list[float]:
        """Server-side latency: per-request guard time inside the serving container."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(concurrency) as ex:
            return list(ex.map(lambda t: self.service.score(t).latency_ms, texts))
