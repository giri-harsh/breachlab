"""Real ONNX Runtime + tokenizers backend for GuardModel (CPU).

Two graph flavours share one loader: the plain slim graph (``input_ids``, ``attention_mask`` -> 4 logits) and the
prefix-KV graph (``guard-kv.onnx``), which additionally takes the precomputed K/V of the fixed prompt prefix and
therefore only processes the message and the suffix (ADR-0039 rung 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from breachlab.common.config import ServeConfig
from breachlab.serve.guard_runtime import GuardModel, GuardParams

KV_FILE = "guard-kv.onnx"


@dataclass(frozen=True)
class Prefix:
    """Precomputed K/V of one fixed prompt prefix (input or output guard)."""

    ids: tuple[int, ...]
    k: NDArray[np.float32]
    v: NDArray[np.float32]


def load_prefixes(model_dir: Path) -> list[Prefix]:
    out: list[Prefix] = []
    for f in sorted(model_dir.glob("prefix_kv_*.npz")):
        with np.load(f) as z:
            out.append(Prefix(tuple(int(i) for i in z["ids"]), z["k"], z["v"]))
    if not out:
        raise FileNotFoundError(f"{KV_FILE} needs prefix_kv_*.npz next to it in {model_dir}")
    return out


def match_prefix(ids: list[int], prefixes: list[Prefix]) -> tuple[list[int], Prefix]:
    """Split ``ids`` into (tokens after the prefix, the matching prefix). The prefix is a hard requirement."""
    for p in prefixes:
        n = len(p.ids)
        if tuple(ids[:n]) == p.ids:
            return ids[n:], p
    raise ValueError("prompt does not start with a known prefix")


def load_guard(
    model_dir: Path,
    serve: ServeConfig,
    version: str,
    threads: int = 8,
    file: str = "guard-int8.onnx",
    model_bytes: bytes | None = None,
    kv: bool | None = None,
) -> GuardModel:  # pragma: no cover - needs onnxruntime + weights
    """``model_bytes``: the already-downloaded ONNX file (single-file graphs only); the session is built from RAM
    instead of ``model_dir / file`` (serverless host without a persistent disk, ADR-0080). The tokenizer,
    calibration and prefix files still come from ``model_dir``. ``kv`` marks a prefix-KV graph whose file name is
    not ``guard-kv.onnx`` (default: decided by the name)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        model_bytes if model_bytes is not None else str(model_dir / file),
        opts,
        providers=["CPUExecutionProvider"],
    )
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    cal_path = model_dir / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    params = GuardParams(
        version, float(cal.get("temperature", 1.0)), float(cal.get("threshold", 0.5))
    )
    prefixes = load_prefixes(model_dir) if (file == KV_FILE if kv is None else kv) else None

    def encode(text: str) -> list[int]:
        return list(tok.encode(text, add_special_tokens=False).ids)

    def logits(ids: list[int]) -> NDArray[np.float64]:
        if prefixes is None:
            arr = np.asarray([ids], dtype=np.int64)
            out = sess.run(None, {"input_ids": arr, "attention_mask": np.ones_like(arr)})[0]
        else:
            tail, p = match_prefix(ids, prefixes)
            arr = np.asarray([tail], dtype=np.int64)
            feeds = {
                "input_ids": arr,
                "attention_mask": np.ones_like(arr),
                "past_k": p.k,
                "past_v": p.v,
            }
            out = sess.run(None, feeds)[0]
        return np.asarray(out[0], dtype=np.float64)

    return GuardModel(encode, logits, serve.words(), params, serve.max_tokens.model_dump())
