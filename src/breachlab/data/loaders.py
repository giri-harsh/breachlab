"""Dataset loaders (one per ADR-0013 dataset). Each pins a revision and checks its license at load."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from breachlab.data.schema import Category, Label, Row

ALLOWED_LICENSES = {"apache-2.0", "mit", "odc-by", "cc-by-4.0", "cc-by-sa-3.0"}
DENIED_LICENSES = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "other"}


class LicenseError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    revision: str
    license: str
    files: tuple[str, ...]
    to_row: Callable[[dict[str, Any]], tuple[str, Label, Category, str] | None]
    gated: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.repo_id


def check_license(spec: DatasetSpec, hub_license: str | None = None) -> None:
    """Raise unless the pinned license is allowed and (if given) matches the live hub license."""
    lic = spec.license.lower()
    if lic in DENIED_LICENSES or lic not in ALLOWED_LICENSES:
        raise LicenseError(f"{spec.repo_id}: license {lic!r} not permitted")
    if hub_license is not None and hub_license.lower() != lic:
        raise LicenseError(f"{spec.repo_id}: pinned {lic!r} but hub reports {hub_license!r}")


def _deepset(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("text", "")).strip()
    if not t:
        return None
    return (
        (t, "UNSAFE", "injection", "deepset")
        if int(rec["label"]) == 1
        else (t, "SAFE", "none", "benign")
    )


def _jackhhao(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("prompt", "")).strip()
    if not t:
        return None
    return (
        (t, "UNSAFE", "jailbreak", "jackhhao")
        if rec.get("type") == "jailbreak"
        else (t, "SAFE", "none", "benign")
    )


def _gandalf(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("text", "")).strip()
    return (t, "UNSAFE", "exfiltration", "gandalf") if t else None


def _wildjb(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("adversarial") or rec.get("vanilla") or "").strip()
    if not t:
        return None
    data_type = str(rec.get("data_type", ""))
    if data_type == "adversarial_harmful":
        return (t, "UNSAFE", "jailbreak", "wildjb")
    if data_type == "adversarial_benign":
        return (t, "SAFE", "none", "hard_benign_wild")
    return None


def _jbb(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("Goal", "")).strip()
    return (t, "UNSAFE", "jailbreak", "jbb") if t else None


def _ultrachat(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("prompt", "")).strip()
    return (t[:500], "SAFE", "none", "benign") if t else None


def _dolly(rec: dict[str, Any]) -> tuple[str, Label, Category, str] | None:
    t = str(rec.get("instruction", "")).strip()
    return (t, "SAFE", "none", "benign") if t else None


SPECS: dict[str, DatasetSpec] = {
    s.repo_id: s
    for s in [
        DatasetSpec(
            "deepset/prompt-injections",
            "4f61ecb038e9c3fb77e21034b22511b523772cdd",
            "apache-2.0",
            (
                "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                "data/test-00000-of-00001-701d16158af87368.parquet",
            ),
            _deepset,
        ),
        DatasetSpec(
            "jackhhao/jailbreak-classification",
            "2f2ceeb39658696fd3f462403562b6eea5306287",
            "apache-2.0",
            ("default/jailbreak_dataset_full.csv",),
            _jackhhao,
        ),
        DatasetSpec(
            "Lakera/gandalf_ignore_instructions",
            "04737b65e90a6794ec227012e4a255a7def6344b",
            "mit",
            ("data/train-00000-of-00001-ded53be747ff55cd.parquet",),
            _gandalf,
        ),
        DatasetSpec(
            "allenai/wildjailbreak",
            "5ddc12a7894f842b0619b8e1c7ee496b198af009",
            "odc-by",
            ("train/train.tsv",),
            _wildjb,
            gated=True,
        ),
        DatasetSpec(
            "JailbreakBench/JBB-Behaviors",
            "886acc352a31533ffbcf4ef22c744658688086fc",
            "mit",
            ("data/harmful-behaviors.csv",),
            _jbb,
        ),
        DatasetSpec(
            "HuggingFaceH4/ultrachat_200k",
            "8049631c405ae6576f93f445c6b8166f76f5505a",
            "mit",
            ("data/train_sft-00000-of-00003-a3ecf92756993583.parquet",),
            _ultrachat,
        ),
        DatasetSpec(
            "databricks/databricks-dolly-15k",
            "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
            "cc-by-sa-3.0",
            ("databricks-dolly-15k.jsonl",),
            _dolly,
        ),
    ]
}


def records_to_rows(
    spec: DatasetSpec, records: Iterable[dict[str, Any]], limit: int | None = None
) -> list[Row]:
    """Convert raw records to Rows after the license check. Pure; used by tests and Modal build."""
    check_license(spec)
    rows: list[Row] = []
    for rec in records:
        conv = spec.to_row(rec)
        if conv is None:
            continue
        text, label, cat, family = conv
        rows.append(Row.new(text, label, cat, spec.source, attack_family=family))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def parse_bytes(name: str, data: bytes) -> Iterator[dict[str, Any]]:
    """Parse a downloaded dataset file (csv/tsv/jsonl/parquet) into dict records."""
    if name.endswith(".jsonl"):
        for line in data.decode("utf-8").split("\n"):
            if line.strip():
                yield json.loads(line)
    elif name.endswith((".csv", ".tsv")):
        delim = "\t" if name.endswith(".tsv") else ","
        yield from csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter=delim)
    elif name.endswith(".parquet"):  # pragma: no cover - needs pandas+pyarrow
        import pandas as pd

        yield from pd.read_parquet(io.BytesIO(data)).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported file type: {name}")


def fetch_rows(
    spec: DatasetSpec, limit: int | None = None, token: str | None = None
) -> list[Row]:  # pragma: no cover
    """Download dataset files (datasets only, never model weights) at the pinned revision."""
    import httpx

    check_license(spec)
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    recs: list[dict[str, Any]] = []
    for f in spec.files:
        url = f"https://huggingface.co/datasets/{spec.repo_id}/resolve/{spec.revision}/{f}"
        r = httpx.get(url, headers=hdr, follow_redirects=True, timeout=120)
        r.raise_for_status()
        recs.extend(parse_bytes(f, r.content))
    return records_to_rows(spec, recs, limit)
