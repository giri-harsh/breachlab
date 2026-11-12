"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal hello-world: CPU and GPU (logs nvidia-smi to MLflow). Entry points are not unit-tested.

GPU type comes from BREACHLAB_GPU (default L4; "none" skips the GPU function, e.g. when the
Modal workspace has no payment method and every GPU function is rejected at app start).
"""

import os
import subprocess

import modal

GPU = os.environ.get("BREACHLAB_GPU", "L4")
app = modal.App("breachlab-hello")
image = modal.Image.debian_slim(python_version="3.11").pip_install("mlflow-skinny")


@app.function(image=image)
def hello_cpu() -> str:
    return "hello from modal cpu"


if GPU != "none":

    @app.function(
        image=image,
        gpu=GPU,
        timeout=300,
        secrets=[
            modal.Secret.from_local_environ(
                ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
            )
        ],
    )
    def hello_gpu() -> str:
        import mlflow

        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
        mlflow.set_experiment("breachlab-phase1")
        with mlflow.start_run(run_name=f"modal-gpu-hello-{GPU}"):
            mlflow.log_text(out, "nvidia-smi.txt")
            mlflow.log_param("gpu", GPU)
        return out


@app.local_entrypoint()
def main() -> None:
    print(hello_cpu.remote())
    if GPU != "none":
        print(hello_gpu.remote())  # pragma: no cover
