"""OPTIONAL, UNUSED (ADR-0037: Modal is permanently out of the plan). Modal CPU ops: alembic migration against Neon (the sandbox cannot reach Postgres TCP)."""

import modal

app = modal.App("breachlab-ops")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sqlalchemy>=2", "alembic", "psycopg[binary]>=3.1", "pydantic>=2.7")
    .add_local_python_source("breachlab")
    .add_local_file("alembic.ini", "/app/alembic.ini")
    .add_local_dir("migrations", "/app/migrations")
)


@app.function(
    image=image, secrets=[modal.Secret.from_local_environ(["NEON_DATABASE_URL"])], timeout=300
)
def migrate() -> str:
    import subprocess

    out = subprocess.run(
        ["alembic", "upgrade", "head"], cwd="/app", capture_output=True, text=True, check=False
    )
    cur = subprocess.run(
        ["alembic", "current"], cwd="/app", capture_output=True, text=True, check=False
    )
    if out.returncode:
        raise RuntimeError(out.stderr[-1500:])
    return (out.stdout + out.stderr + cur.stdout)[-1500:]


@app.local_entrypoint()
def main() -> None:
    print(migrate.remote())
