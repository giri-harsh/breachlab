"""Stage 3: DPO pairs from mined rows, then a short DPO pass from the SFT adapter (PRD 7.4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from breachlab.common.config import ServeConfig, TrainConfig
from breachlab.data.schema import Row
from breachlab.train.mine import Mined
from breachlab.train.prompts import label_index, render_prompt


@dataclass(frozen=True)
class Pair:
    prompt: str
    chosen: str
    rejected: str


def build_pairs(mined: list[Mined], serve: ServeConfig) -> list[Pair]:
    """chosen = correct verbalizer word; rejected = the model's wrong top answer (or runner-up for near misses)."""
    words = serve.words()
    pairs: list[Pair] = []
    for m in mined:
        gold = label_index(m.row)
        rejected = m.runner_up if m.kind == "near_miss" else m.top
        if rejected == gold:
            continue
        pairs.append(Pair(render_prompt(m.row.text, words), words[gold], words[rejected]))
    return pairs


def balance_pairs(pairs: list[Pair], max_pairs: int, seed: int = 0) -> list[Pair]:
    rng = np.random.default_rng(seed)
    if len(pairs) <= max_pairs:
        return pairs
    idx = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[i] for i in sorted(idx)]


def run_dpo(
    cfg: TrainConfig, serve: ServeConfig, sft_adapter: Path, pairs: list[Pair]
) -> Path:  # pragma: no cover - GPU
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.bfloat16 if cfg.sft.precision == "bf16" else torch.float16
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
    model = PeftModel.from_pretrained(base, str(sft_adapter), is_trainable=True)
    out = cfg.output_dir / f"dpo-{cfg.short_name}-s{cfg.seed}"
    args = DPOConfig(
        output_dir=str(out / "ckpt"),
        beta=cfg.dpo.beta,
        num_train_epochs=cfg.dpo.epochs,
        learning_rate=cfg.dpo.lr,
        per_device_train_batch_size=cfg.dpo.per_device_batch,
        gradient_accumulation_steps=cfg.dpo.grad_accum,
        fp16=cfg.sft.precision == "fp16",
        bf16=cfg.sft.precision == "bf16",
        max_length=cfg.max_len + 8,
        seed=cfg.seed,
        report_to=[],
        save_strategy="no",
    )
    ds = Dataset.from_list([p.__dict__ for p in pairs])
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    return adapter


def pairs_from_rows(rows: list[Row], mined: list[Mined], serve: ServeConfig) -> list[Pair]:
    """Convenience for tests/CLI: pairs only for rows present in ``rows``."""
    ids = {r.id for r in rows}
    return build_pairs([m for m in mined if m.row.id in ids], serve)
