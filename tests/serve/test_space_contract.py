"""The shipped service (``space_app``) against the frozen guard contract (docs/CONTRACTS.md s1).

The same assertions that ``tests/gateway/test_guard_contract.py`` runs against a deployment (``GUARD_URL``) run
here against the real FastAPI app with fake ONNX sessions, so a host change cannot silently break the contract.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from breachlab.serve.guard_runtime import GuardModel, GuardParams
from breachlab.serve.space_app import Settings, create_app

# The contract tests themselves; imported names are collected here and use this module's ``client`` fixture.
from tests.gateway.test_guard_contract import (  # noqa: F401
    test_aliases_and_fallback,
    test_attack_and_benign_are_separated,
    test_health,
    test_invalid_requests_are_client_errors,
    test_score_ok_shapes,
)
from tests.serve.test_space_app import WORDS, enc, make_dirs

MAXTOK = {"input_guard": 200, "output_filter": 200}


def _loader(directory: Path, version: str, variant: str, threads: int, alias: str) -> GuardModel:
    def logits(ids: list[int]) -> Any:
        text = "".join(chr(i) for i in ids).split("<|im_start|>user\n")[-1].lower()
        out = np.zeros(4)
        out[1 if "ignore all previous" in text else 0] = 6.0
        return out

    return GuardModel(enc, logits, WORDS, GuardParams(version, 1.0, 0.5), MAXTOK)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[httpx.Client]:
    app = create_app(Settings(model_dir=make_dirs(tmp_path)), loader=_loader, background=False)
    with TestClient(app) as c:
        yield c  # type: ignore[misc]


def test_empty_reply_is_scored_but_empty_message_is_rejected(client: httpx.Client) -> None:
    assert client.post("/score", json={"text": "  ", "kind": "input"}).status_code == 422
    r = client.post("/score", json={"text": "", "kind": "output"})
    assert r.status_code == 200 and r.json()["label"] == "SAFE"
