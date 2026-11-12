"""Score rows with a (peft) model: logits over the 4 verbalizer tokens. GPU only; pure parts tested."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.common.config import ServeConfig
from breachlab.data.schema import Row
from breachlab.train.prompts import encode_prompt


def batches(n: int, size: int) -> list[range]:
    return [range(i, min(i + size, n)) for i in range(0, n, size)]


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad so position -1 is the answer position for every item."""
    n = max(len(s) for s in seqs)
    ids = [[pad_id] * (n - len(s)) + s for s in seqs]
    mask = [[0] * (n - len(s)) + [1] * len(s) for s in seqs]
    return ids, mask


def score_rows(  # pragma: no cover - GPU
    model: Any, tok: Any, rows: list[Row], serve: ServeConfig, max_len: int = 512, bs: int = 16
) -> NDArray[np.float64]:
    """(n, 4) verbalizer logits in label order, one forward pass per batch."""
    import torch

    words, ids4 = serve.words(), serve.token_ids()

    def enc(s: str) -> list[int]:
        return list(tok.encode(s, add_special_tokens=False))

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = np.zeros((len(rows), 4))
    model.eval()
    for rng in batches(len(rows), bs):
        seqs = [encode_prompt(enc, rows[i].text, words, "input", max_len) for i in rng]
        ids, mask = left_pad(seqs, pad)
        m = torch.tensor(mask).cuda()
        with torch.no_grad():
            logits = model(
                input_ids=torch.tensor(ids).cuda(),
                attention_mask=m,
                position_ids=(m.cumsum(-1) - 1).clamp(min=0),
                logits_to_keep=1,
            ).logits[:, -1, :]
        out[rng.start : rng.stop] = logits[:, ids4].float().cpu().numpy()
    return out
