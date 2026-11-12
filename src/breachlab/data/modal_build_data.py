"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal CPU entrypoint for the Data v1 build (detached, resumable shards in a Volume). Not run in CI.

Launch:  modal run --detach src/breachlab/data/modal_build_data.py
Fetch:   modal volume get breachlab-data v1 data/processed/v1
"""

import modal

app = modal.App("breachlab-build-data")
vol = modal.Volume.from_name("breachlab-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "pydantic>=2.7",
        "httpx",
        "pandas",
        "pandera",
        "datasketch",
        "pyarrow",
        "pyyaml",
        "mlflow-skinny",
    )
    .add_local_python_source("breachlab")
)
SECRETS = [
    modal.Secret.from_local_environ(
        [
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "HF_TOKEN",
            "MLFLOW_TRACKING_URI",
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD",
        ]
    )
]
CPU_USD_PER_CORE_HOUR = 0.0473  # Modal published CPU price; used only for the cost estimate
CORES = 4

SEED_LIMIT = 6000  # rows kept per source dataset


def projected_cost_usd(hours: float, cores: int = CORES) -> float:
    return round(hours * cores * CPU_USD_PER_CORE_HOUR, 2)


@app.function(
    image=image,
    volumes={"/data": vol},
    secrets=SECRETS,
    timeout=8 * 3600,
    cpu=CORES,
    memory=8192,
)
def build_v1(
    version: str = "v1",
    llm_parent_limit: int = 2500,
    k_det: int = 2,
    hb_llm_calls: int = 150,
    workers: int = 8,
    balance: bool = False,
    max_llm_shards: int | None = None,
) -> str:
    import json
    import os
    import threading
    import time
    from pathlib import Path

    import httpx
    import mlflow

    from breachlab.data.build import build
    from breachlab.data.llm import GeminiPool, GroqClient, LockedRoundRobin
    from breachlab.data.loaders import SPECS, fetch_rows

    t0 = time.time()
    out = Path(f"/data/{version}")
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def committer() -> None:  # persist shards so a crash resumes from the last commit
        while not stop.wait(60):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()

    token = os.environ.get("HF_TOKEN")
    seeds = []
    skipped: dict[str, str] = {}
    cache = out / "seeds.jsonl"
    if cache.exists():
        from breachlab.data.schema import Row, jsonl_lines

        seeds = [Row.model_validate_json(x) for x in jsonl_lines(cache.read_text("utf-8"))]
    else:
        for spec in SPECS.values():
            try:
                seeds += fetch_rows(spec, limit=SEED_LIMIT, token=token)
            except httpx.HTTPStatusError as exc:
                if spec.gated and exc.response.status_code in (401, 403):
                    skipped[spec.repo_id] = f"gated, HTTP {exc.response.status_code} (ADR-0019)"
                    continue
                raise
        cache.write_text("\n".join(r.model_dump_json() for r in seeds) + "\n", encoding="utf-8")
        (out / "skipped_sources.json").write_text(json.dumps(skipped), encoding="utf-8")

    attack_llm = LockedRoundRobin(
        [
            GeminiPool(),
            GeminiPool(),
            GroqClient(retries=2),
            GroqClient("qwen/qwen3.8-27b", retries=2),
        ]
    )
    rows = build(
        seeds,
        attack_llm,
        out,
        version=version,
        n_hard_benign=100,
        batch=True,
        k_det=k_det,
        workers=workers,
        llm_parent_limit=llm_parent_limit,
        hb_llm_calls=hb_llm_calls,
        hb_llm=LockedRoundRobin(
            [GroqClient(retries=2), GeminiPool(), GroqClient("openai/gpt-oss-20b", retries=2)],
            rounds=60,
            pause=30,
        ),
        balance=balance,
        max_llm_shards=max_llm_shards,
    )
    stop.set()
    vol.commit()
    hours = (time.time() - t0) / 3600
    mlflow.set_experiment("breachlab-data")
    with mlflow.start_run(run_name=f"build-data-{version}"):
        mlflow.log_params({"version": version, "skipped_sources": json.dumps(skipped)})
        mlflow.log_metrics(
            {"rows": float(len(rows)), "actual_usd": projected_cost_usd(hours), "hours": hours}
        )
    return f"built {len(rows)} rows in {hours:.2f} h"


@app.local_entrypoint()
def main(version: str = "v1", balance: bool = False, max_llm_shards: int = -1) -> None:
    import mlflow

    proj = projected_cost_usd(1.5)
    mlflow.set_experiment("breachlab-data")
    with mlflow.start_run(run_name=f"projected-cost-data-{version}"):
        mlflow.log_metric("projected_usd", proj)
    assert proj <= 25, "projected cost exceeds per-job cap"
    call = build_v1.spawn(
        version=version,
        balance=balance,
        max_llm_shards=None if max_llm_shards < 0 else max_llm_shards,
    )
    print("function call id:", call.object_id)
