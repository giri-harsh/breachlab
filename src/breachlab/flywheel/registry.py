"""MLflow registry aliases (PRD 10): register, promote challenger -> champion, rollback.

``champion`` is what the gateway serves; ``previous`` remembers the last champion so a rollback is a single
alias move (takes effect when the gateway next polls the alias, every 60 s). ``client`` is an
``mlflow.MlflowClient`` or any object with the same three methods (tests use a fake).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol

MODEL = "breachlab-guard"


class RegistryClient(Protocol):
    def get_model_version_by_alias(self, name: str, alias: str) -> Any: ...
    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None: ...
    def delete_registered_model_alias(self, name: str, alias: str) -> None: ...


@dataclass(frozen=True)
class Move:
    action: str
    version: str
    previous: str | None
    seconds: float


def _get(client: RegistryClient, alias: str, name: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except Exception:  # noqa: BLE001 - mlflow raises RestException when the alias is unset
        return None


def promote(client: RegistryClient, version: str | None = None, name: str = MODEL) -> Move:
    """Move ``champion`` to ``version`` (default: the current ``challenger``); remember the old champion."""
    t0 = time.perf_counter()
    target = version or _get(client, "challenger", name)
    if target is None:
        raise ValueError("no version given and no challenger alias set")
    old = _get(client, "champion", name)
    if old == target:
        raise ValueError(f"version {target} is already champion")
    if old is not None:
        client.set_registered_model_alias(name, "previous", old)
    client.set_registered_model_alias(name, "champion", target)
    return Move("promote", target, old, time.perf_counter() - t0)


def rollback(client: RegistryClient, name: str = MODEL) -> Move:
    """Move ``champion`` back to ``previous`` (and keep the rolled-back version as ``previous``)."""
    t0 = time.perf_counter()
    prev, cur = _get(client, "previous", name), _get(client, "champion", name)
    if prev is None:
        raise ValueError("no previous champion to roll back to")
    client.set_registered_model_alias(name, "champion", prev)
    if cur is not None:
        client.set_registered_model_alias(name, "previous", cur)
    return Move("rollback", prev, cur, time.perf_counter() - t0)


def main(argv: list[str]) -> int:  # pragma: no cover - needs MLflow
    import mlflow

    client = mlflow.MlflowClient()
    cmd = argv[0] if argv else ""
    if cmd == "promote":
        m = promote(client, argv[1] if len(argv) > 1 else None)
    elif cmd == "rollback":
        m = rollback(client)
    else:
        print("usage: registry promote [version] | rollback", file=sys.stderr)
        return 2
    print(f"{m.action}: champion={m.version} previous={m.previous} in {m.seconds * 1000:.0f} ms")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
