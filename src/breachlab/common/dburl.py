"""Database URL helpers shared by alembic and the app settings."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def normalize_db_url(url: str, *, strip_channel_binding: bool = False) -> str:
    """Return a SQLAlchemy psycopg URL; optionally drop the ``channel_binding`` query param.

    The env value itself is never edited: callers pass the stripped copy to the driver only.
    """
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    if not strip_channel_binding:
        return url
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "channel_binding"]
    return urlunsplit(parts._replace(query=urlencode(query)))
