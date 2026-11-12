"""ASGI entry point shared by Docker, Modal and Vercel: ``app`` is built lazily from the environment."""

from __future__ import annotations

from breachlab.gateway.app import create_app

app = create_app()
