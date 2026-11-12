import os

from alembic import context
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from breachlab.common.db import Base
from breachlab.common.dburl import normalize_db_url

target_metadata = Base.metadata


def _raw_url() -> str:
    url = os.environ.get("NEON_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("NEON_DATABASE_URL (or DATABASE_URL) not set")
    return url


def _engine():  # type: ignore[no-untyped-def]
    """Connect as-is; if the driver rejects channel_binding, retry with it stripped (ADR-0017)."""
    raw = _raw_url()
    engine = create_engine(normalize_db_url(raw))
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
    except OperationalError as exc:
        if "channel_binding" not in str(exc):
            raise
        engine = create_engine(normalize_db_url(raw, strip_channel_binding=True))
    return engine


def run() -> None:
    with _engine().connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    context.configure(
        url=normalize_db_url(_raw_url()), target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    run()
