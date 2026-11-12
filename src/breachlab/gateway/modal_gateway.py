"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal ASGI deployment of the gateway (PRD 4.1). Deploy: modal deploy src/breachlab/gateway/modal_gateway.py"""

import modal

app = modal.App("breachlab-gateway")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "sqlalchemy>=2",
        "psycopg[binary]>=3.1",
        "pydantic>=2.7",
        "pydantic-settings",
        "pyyaml",
        "httpx",
        "numpy",
    )
    .env({"BREACHLAB_CONFIG_DIR": "/app/configs"})
    .add_local_python_source("breachlab")
    .add_local_dir("configs", "/app/configs")
)
SECRETS = [
    modal.Secret.from_local_environ(
        [
            "NEON_DATABASE_URL",
            "GROQ_API_KEY",
            "ADMIN_PASSWORD",
            "GUARD_URL",
            "CORS_ORIGIN",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
        ]
    )
]


@app.function(image=image, secrets=SECRETS, min_containers=0, scaledown_window=600)
@modal.asgi_app()
def web():  # type: ignore[no-untyped-def]
    from breachlab.gateway.asgi import app as fastapi_app

    return fastapi_app
