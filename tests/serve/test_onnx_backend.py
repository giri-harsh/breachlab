from pathlib import Path

import numpy as np
import pytest

from breachlab.serve import onnx_backend, space_app
from breachlab.train import export_kv, export_onnx


def prefix(ids: list[int], layers: int = 2) -> onnx_backend.Prefix:
    k = np.zeros((layers, 1, 2, len(ids), 4), dtype=np.float32)
    return onnx_backend.Prefix(tuple(ids), k, k.copy())


def test_match_prefix_splits_and_rejects() -> None:
    a, b = prefix([1, 2, 3]), prefix([9, 8])
    tail, p = onnx_backend.match_prefix([9, 8, 5, 6], [a, b])
    assert tail == [5, 6] and p is b
    tail, p = onnx_backend.match_prefix([1, 2, 3], [a, b])
    assert tail == [] and p is a
    with pytest.raises(ValueError):
        onnx_backend.match_prefix([7, 7, 7], [a, b])


def test_pack_and_load_prefixes_roundtrip(tmp_path: Path) -> None:
    k = np.random.default_rng(0).normal(size=(3, 1, 2, 4, 5)).astype(np.float32)
    v = k + 1
    np.savez(tmp_path / export_kv.prefix_file("input"), **export_kv.pack_prefix([5, 6, 7, 8], k, v))
    np.savez(
        tmp_path / export_kv.prefix_file("output"), **export_kv.pack_prefix([5, 6, 7, 9], k, v)
    )
    loaded = onnx_backend.load_prefixes(tmp_path)
    assert len(loaded) == 2 and {p.ids[-1] for p in loaded} == {8, 9}
    assert np.array_equal(loaded[0].k, k) and np.array_equal(loaded[0].v, v)


def test_load_prefixes_missing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        onnx_backend.load_prefixes(tmp_path)


def test_pack_prefix_validates_shapes() -> None:
    k = np.zeros((2, 1, 2, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        export_kv.pack_prefix([1, 2, 3], k, k)  # 3 ids vs 4 positions
    with pytest.raises(ValueError):
        export_kv.pack_prefix([1, 2, 3, 4], k, k[:1])
    with pytest.raises(ValueError):
        export_kv.pack_prefix([1, 2, 3, 4], k[:, 0], k[:, 0])


def test_variant_tables_agree() -> None:
    """The Space's file table and the exporter's must name the same files."""
    assert space_app.VARIANT_FILES == export_onnx.VARIANT_FILES
    assert export_onnx.VARIANT_FILES["kv"] == export_kv.KV_FILE == onnx_backend.KV_FILE
