"""Stage 1: LoRA SFT with loss on the single answer token only (PRD 7.4.1)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breachlab.common.config import ServeConfig, TrainConfig
from breachlab.data.schema import Row
from breachlab.train.prompts import encode_prompt, label_index


@dataclass(frozen=True)
class Example:
    text: str
    words: tuple[str, ...]
    target_id: int
    label_idx: int


def build_examples(rows: list[Row], serve: ServeConfig) -> list[Example]:
    """One example per row: chat prompt + the verbalizer token id of its class."""
    words, ids = serve.words(), serve.token_ids()
    return [Example(r.text, tuple(words), ids[label_index(r)], label_index(r)) for r in rows]


def encode_example(ex: Example, tokenize: Any, max_len: int) -> dict[str, Any]:
    """Prompt ids (truncated head+tail) and the answer token id; the loss is taken only at the last prompt
    position (left-padded batches), so full-vocab logits are never materialised for the whole sequence."""
    ids = encode_prompt(tokenize, ex.text, ex.words, "input", max_len)
    return {"input_ids": ids, "target": ex.target_id}


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, list[Any]]:
    """Left-pad so position -1 is the answer position for every item."""
    n = max(len(b["input_ids"]) for b in batch)
    out: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = n - len(b["input_ids"])
        out["input_ids"].append([pad_id] * pad + b["input_ids"])
        out["attention_mask"].append([0] * pad + [1] * len(b["input_ids"]))
        out["labels"].append(b["target"])
    return out


def warmup_steps(n_rows: int, cfg: TrainConfig) -> int:
    """Integer warmup steps (3% of total): ``warmup_ratio`` was removed in recent transformers."""
    steps = -(-n_rows // cfg.sft.effective_batch) * cfg.sft.epochs
    return max(1, round(cfg.sft.warmup_ratio * steps))


def gpu_usd(seconds: float, usd_per_hour: float) -> float:
    return round(seconds / 3600 * usd_per_hour, 4)


def run_sft(
    cfg: TrainConfig, serve: ServeConfig, train_rows: list[Row], val_rows: list[Row]
) -> Path:  # pragma: no cover - GPU
    """Train the LoRA adapter on a GPU box (Kaggle T4 / Modal). Returns the adapter directory."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.bfloat16 if cfg.sft.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lc = cfg.sft.lora
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc.r,
            lora_alpha=lc.alpha,
            lora_dropout=lc.dropout,
            target_modules=lc.target_modules,
            task_type="CAUSAL_LM",
        ),
    )
    for p in model.parameters():  # LoRA params in fp32 for stable fp16 training
        if p.requires_grad:
            p.data = p.data.float()

    def tokenize(s: str) -> list[int]:
        return list(tok.encode(s, add_special_tokens=False))

    class DS(torch.utils.data.Dataset):  # type: ignore[misc]
        def __init__(self, rows: list[Row]) -> None:
            self.items = [
                encode_example(e, tokenize, cfg.max_len) for e in build_examples(rows, serve)
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> dict[str, Any]:
            return self.items[i]

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {k: torch.tensor(v) for k, v in collate(batch, pad).items()}

    class AnswerTokenTrainer(Trainer):  # type: ignore[misc]
        def compute_loss(
            self, model: Any, inputs: Any, return_outputs: bool = False, **kw: Any
        ) -> Any:
            m = inputs["attention_mask"]
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=m,
                position_ids=(m.cumsum(-1) - 1).clamp(min=0),
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :].float()
            loss = torch.nn.functional.cross_entropy(logits, inputs["labels"])
            return (loss, out) if return_outputs else loss

    out = cfg.output_dir / f"sft-{cfg.short_name}-s{cfg.seed}"
    args = TrainingArguments(
        output_dir=str(out / "ckpt"),
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.lr,
        lr_scheduler_type=cfg.sft.scheduler,
        warmup_steps=warmup_steps(len(train_rows), cfg),
        per_device_train_batch_size=cfg.sft.per_device_batch,
        label_names=["labels"],
        prediction_loss_only=True,
        per_device_eval_batch_size=cfg.sft.per_device_batch,
        gradient_accumulation_steps=cfg.sft.grad_accum,
        fp16=cfg.sft.precision == "fp16",
        bf16=cfg.sft.precision == "bf16",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        seed=cfg.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    t0 = time.time()
    trainer = AnswerTokenTrainer(
        model=model,
        args=args,
        train_dataset=DS(train_rows),
        eval_dataset=DS(val_rows),
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.sft.early_stop_patience)],
    )
    resume = any((out / "ckpt").glob("checkpoint-*")) if (out / "ckpt").exists() else False
    trainer.train(resume_from_checkpoint=resume or None)
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    (out / "train_seconds.txt").write_text(str(time.time() - t0), encoding="utf-8")
    return adapter
