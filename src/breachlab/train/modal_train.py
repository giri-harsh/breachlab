"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal entrypoints for training. GPU functions exist only when BREACHLAB_GPU is set to a real GPU
(Modal rejects GPU functions without a payment method, ADR-0020); the CPU relay always exists.

Relay: pull a finished Kaggle kernel's output on a CPU box (no weights in the dev container), log it to
MLflow with GPU-seconds/USD, upload adapters to HF, and register the version under alias ``challenger``.
Run: modal run src/breachlab/train/modal_train.py::relay --kernel giriharsh/breachlab-sft-06b-s0 --name guard-v1-06b-s0
"""

import os

import modal

GPU = os.environ.get("BREACHLAB_GPU", "none")
app = modal.App("breachlab-train")
vol = modal.Volume.from_name("breachlab-train", create_if_missing=True)
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle>=1.7.4", "mlflow-skinny", "huggingface_hub", "numpy", "pydantic>=2.7")
    .add_local_python_source("breachlab")
)
SECRETS = [
    modal.Secret.from_local_environ(
        [
            "KAGGLE_API_TOKEN",
            "HF_TOKEN",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        ]
    )
]


@app.function(image=cpu_image, secrets=SECRETS, timeout=3600, volumes={"/vol": vol})
def relay(
    kernel: str, name: str, hf_repo: str = "", projected_usd: float = 0.0
) -> dict[str, object]:
    import json
    import subprocess
    import sys
    from pathlib import Path

    import mlflow

    out = Path("/vol") / name
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", kernel, "-p", str(out)], check=True
    )
    results = next(out.rglob("results.json"))
    res = json.loads(results.read_text())
    mlflow.set_experiment("breachlab-train")
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({"kernel": kernel, "base_model": res["base_model"], "seed": res["seed"]})
        mlflow.set_tags(res.get("repro", {}))  # PRD 12: git SHA, config hash, data version, seed
        mlflow.log_metrics(
            {
                "gpu_seconds": float(res["gpu_seconds"]),
                "gpu_usd": float(res["gpu_usd"]),
                "projected_usd": projected_usd,
            }
        )
        mlflow.log_dict(res, "results.json")
        adapter_dir = results.parent
        if hf_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hf_repo, private=True, exist_ok=True)
            for sub in ("adapter",):
                for d in adapter_dir.rglob(sub):
                    api.upload_folder(
                        folder_path=str(d), repo_id=hf_repo, path_in_repo=d.parent.name
                    )
        reg = mlflow.register_model(f"runs:/{run.info.run_id}/results.json", "breachlab-guard")
        mlflow.tracking.MlflowClient().set_registered_model_alias(
            "breachlab-guard", "challenger", reg.version
        )
    vol.commit()
    return {"run_id": run.info.run_id, "version": reg.version}


if GPU != "none":
    gpu_image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers>=4.51",
            "peft",
            "trl",
            "datasets",
            "accelerate",
            "numpy",
            "pydantic-settings",
            "pyyaml",
            "pydantic>=2.7",
            "pandas",
            "pandera",
            "scipy",
        )
        .add_local_python_source("breachlab")
        .env({"BREACHLAB_CONFIG_DIR": "/app/configs"})
        .add_local_dir("configs", "/app/configs")
    )

    @app.function(image=gpu_image, gpu=GPU, timeout=6 * 3600, volumes={"/vol": vol})
    def train(model: str, seed: int, data_dir: str, no_dpo: bool = False) -> int:
        from breachlab.train.kernel_main import main

        argv = ["--model", model, "--seed", str(seed), "--data-dir", data_dir, "--out", "/vol/out"]
        if no_dpo:
            argv.append("--no-dpo")
        rc = main(argv)
        vol.commit()
        return rc


@app.local_entrypoint()
def main(kernel: str = "", name: str = "guard-run", hf_repo: str = "") -> None:
    if kernel:
        print(relay.remote(kernel, name, hf_repo))
