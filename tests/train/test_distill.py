from __future__ import annotations

import numpy as np

from breachlab.train.distill import CANDIDATES, agreement, pad_batch
from breachlab.train.prompts import wrap_ids


def test_wrap_ids_adds_specials_and_truncates_head_and_tail() -> None:
    assert wrap_ids([1, 2, 3], 10, 100, 101) == [100, 1, 2, 3, 101]
    ids = list(range(50))
    out = wrap_ids(ids, 12, 100, 101)
    assert len(out) == 12 and out[0] == 100 and out[-1] == 101
    assert out[1:6] == [0, 1, 2, 3, 4] and out[6:-1] == [45, 46, 47, 48, 49]


def test_pad_batch_right_pads_with_mask() -> None:
    ids, mask = pad_batch([[1, 2, 3], [4]], 0)
    assert ids == [[1, 2, 3], [4, 0, 0]] and mask == [[1, 1, 1], [1, 0, 0]]


def test_agreement_counts_argmax_and_binary() -> None:
    t = np.array([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 5.0, 0]])
    s = np.array([[5.0, 0, 0, 0], [0, 0, 5.0, 0], [5.0, 0, 0, 0]])
    a = agreement(t, s)
    assert abs(a["argmax4"] - 1 / 3) < 1e-9 and abs(a["binary"] - 2 / 3) < 1e-9


def test_candidates_are_permissively_licensed() -> None:
    assert {c.licence for c in CANDIDATES.values()} <= {"MIT", "Apache-2.0"}
