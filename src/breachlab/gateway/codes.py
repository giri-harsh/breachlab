"""FR-1: per-session, per-level secret generation (server-side only) and hashing."""

from __future__ import annotations

import hashlib
import secrets

# Uppercase letters and digits without ambiguous characters 0 O 1 l I.
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # pragma: allowlist secret
CODE_LENGTH = 8
AMBIGUOUS = set("0O1lI")


def new_secret(length: int = CODE_LENGTH) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def secret_hash(secret: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


def level_secret(key: str, session_id: str, level: int, length: int = CODE_LENGTH) -> str:
    """Deterministic per-session, per-level code: HMAC(server key, session|level) mapped to the alphabet.

    Stateless hosts (serverless) cannot keep plaintext secrets between requests, and the DB stores only the
    hash (PRD 6.1), so the server re-derives the code from its key. The key never leaves the server.
    """
    import hmac

    digest = hmac.new(key.encode(), f"{session_id}|{level}".encode(), hashlib.sha256).digest()
    return "".join(ALPHABET[b % len(ALPHABET)] for b in digest[:length])
