"""Vercel zero-config FastAPI entry for the gateway (fallback host while Modal is unavailable, ADR-0033)."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
os.environ.setdefault("BREACHLAB_CONFIG_DIR", os.path.join(HERE, "configs"))

from breachlab.gateway.asgi import app  # noqa: E402,F401
