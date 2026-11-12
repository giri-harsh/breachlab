"""One training run end to end: SFT -> mine -> DPO -> calibrate, idempotent per stage (PRD 7.4).

Runs on a GPU box (Kaggle T4 kernel, or Modal when a payment method exists). Every stage writes its output
under ``output_dir`` and is skipped on re-run if the output exists, so a restarted kernel resumes. Results
(`results.json`, adapters, logits) are relayed to MLflow / HF by ``breachlab.train.relay`` on a CPU box.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from breachlab.common.config import ServeConfig, TrainConfig
from breachlab.train.calibrate import calibrate, to_preds
from breachlab.train.dpo import build_pairs
from breachlab.train.mine import mine, summary
from breachlab.train.repro import repro_tags
from breachlab.train.scoring import read_rows
from breachlab.train.sft import gpu_usd


def stage_done(path: Path) -> bool:
    return path.exists()


def run(
    cfg: TrainConfig, serve: ServeConfig, *, with_dpo: bool = True
) -> dict[str, Any]:  # pragma: no cover - GPU
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from breachlab.eval.metrics import summarize
    from breachlab.train.dpo import run_dpo
    from breachlab.train.score_model import score_rows
    from breachlab.train.sft import run_sft

    t0 = time.time()
    tag = f"{cfg.short_name}-s{cfg.seed}"
    root = cfg.output_dir / tag
    root.mkdir(parents=True, exist_ok=True)
    train, val = read_rows(cfg.data_dir / "train.jsonl"), read_rows(cfg.data_dir / "val.jsonl")
    pool = read_rows(cfg.data_dir / "mining_pool.jsonl")[: cfg.mining.pool_size]
    res: dict[str, Any] = {"tag": tag, "base_model": cfg.base_model, "seed": cfg.seed}
    res["repro"] = repro_tags(cfg)  # PRD 12: the relay sets these as MLflow run tags

    adapter = cfg.output_dir / f"sft-{cfg.short_name}-s{cfg.seed}" / "adapter"
    if not stage_done(adapter):
        adapter = run_sft(cfg, serve, train, val)
    res["sft_seconds"] = time.time() - t0

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.bfloat16 if cfg.sft.precision == "bf16" else torch.float16

    def load(ad: Path) -> Any:
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
        return PeftModel.from_pretrained(base, str(ad))

    def score_to(ad: Path, rows: Any, name: str) -> np.ndarray:
        f = root / f"{name}.npy"
        if f.exists():
            return np.load(f)  # type: ignore[no-any-return]
        m = load(ad)
        lg = score_rows(m, tok, rows, serve, cfg.max_len)
        np.save(f, lg)
        del m
        torch.cuda.empty_cache()
        return lg

    final = adapter
    val_logits = score_to(adapter, val, "val_sft")
    cal_sft = calibrate(val, val_logits, cfg.calibration.ece_bins, cfg.calibration.max_fpr)
    res["sft"] = {"calibration": cal_sft.__dict__}
    (root / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    if with_dpo:
        try:
            pool_logits = score_to(adapter, pool, "pool_sft")
            mined = mine(pool, pool_logits, cal_sft.temperature, cfg.mining.near_miss_margin)
            res["mining"] = summary(mined)
            pairs = build_pairs(mined, serve)
            res["dpo_pairs"] = len(pairs)
            dpo_adapter = cfg.output_dir / f"dpo-{cfg.short_name}-s{cfg.seed}" / "adapter"
            if pairs and not stage_done(dpo_adapter):
                dpo_adapter = run_dpo(cfg, serve, adapter, pairs)
            if pairs:
                vl = score_to(dpo_adapter, val, "val_dpo")
                cal_dpo = calibrate(val, vl, cfg.calibration.ece_bins, cfg.calibration.max_fpr)
                res["dpo"] = {"calibration": cal_dpo.__dict__}
                sft_f1 = summarize(
                    to_preds(val, val_logits, cal_sft.temperature, cal_sft.threshold)
                )["f1"]
                dpo_f1 = summarize(to_preds(val, vl, cal_dpo.temperature, cal_dpo.threshold))["f1"]
                res["val_f1"] = {"sft": sft_f1, "dpo": dpo_f1}
                if dpo_f1 > sft_f1:
                    final = dpo_adapter
        except Exception as exc:  # noqa: BLE001 - SFT result must survive a DPO failure
            res["dpo_error"] = f"{type(exc).__name__}: {exc}"
    res["final_adapter"] = str(final)
    res["gpu_seconds"] = time.time() - t0
    res["gpu_usd"] = gpu_usd(res["gpu_seconds"], cfg.budget.gpu_usd_per_hour)
    (root / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def run_zeroshot(cfg: TrainConfig, serve: ServeConfig) -> dict[str, Any]:  # pragma: no cover - GPU
    """Zero-shot base-model eval on val (PRD 7.1): same guard prompt, no adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from breachlab.eval.metrics import summarize
    from breachlab.train.score_model import score_rows

    t0 = time.time()
    val = read_rows(cfg.data_dir / "val.jsonl")
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.bfloat16 if cfg.sft.precision == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
    logits = score_rows(model, tok, val, serve, cfg.max_len)
    cal = calibrate(val, logits, cfg.calibration.ece_bins, cfg.calibration.max_fpr)
    res: dict[str, Any] = {
        "tag": f"zeroshot-{cfg.short_name}",
        "base_model": cfg.base_model,
        "seed": cfg.seed,
        "at_0.5": summarize(to_preds(val, logits, 1.0, 0.5)),
        "at_fpr2": summarize(to_preds(val, logits, cal.temperature, cal.threshold)),
        "calibration": cal.__dict__,
        "n_val": len(val),
        "gpu_seconds": time.time() - t0,
    }
    res["gpu_usd"] = gpu_usd(res["gpu_seconds"], cfg.budget.gpu_usd_per_hour)
    root = cfg.output_dir / res["tag"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def run_deberta(cfg: TrainConfig) -> dict[str, Any]:  # pragma: no cover - GPU/CPU kernel
    """protectai/deberta-v3-base-prompt-injection-v2 on val + gold_test (comparison row 3)."""
    import torch
    from transformers import pipeline

    t0 = time.time()
    pipe = pipeline(
        "text-classification",
        model="protectai/deberta-v3-base-prompt-injection-v2",
        truncation=True,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1,
    )
    out: dict[str, Any] = {"model": "protectai/deberta-v3-base-prompt-injection-v2"}
    for split in ("val", "gold_test"):
        path = cfg.data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_rows(path)
        preds = pipe([r.text for r in rows], batch_size=32)
        out[split] = {
            r.id: ["UNSAFE" if p["label"].upper() == "INJECTION" else "SAFE", float(p["score"])]
            for r, p in zip(rows, preds, strict=True)
        }
    out["seconds"] = time.time() - t0
    root = cfg.output_dir / "deberta"
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def run_evalgold(  # pragma: no cover - GPU
    cfg: TrainConfig, serve: ServeConfig, adapters_root: Path
) -> dict[str, Any]:
    """Score val + gold_test with every adapter found under ``adapters_root`` (kernel_sources mount)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from breachlab.train.score_model import score_rows

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    dtype = torch.bfloat16 if cfg.sft.precision == "bf16" else torch.float16
    root = cfg.output_dir / f"evalgold-{cfg.short_name}-s{cfg.seed}"
    root.mkdir(parents=True, exist_ok=True)
    splits = {s: read_rows(cfg.data_dir / f"{s}.jsonl") for s in ("val", "gold_test")}
    for s, rows in splits.items():
        (root / f"{s}_ids.json").write_text(json.dumps([r.id for r in rows]), encoding="utf-8")
    found: dict[str, str] = {}
    zs = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
    for s, rows in splits.items():
        np.save(root / f"{s}_zeroshot.npy", score_rows(zs, tok, rows, serve, cfg.max_len))
    del zs
    torch.cuda.empty_cache()
    for ad in sorted(adapters_root.rglob("adapter")):
        name = ad.parent.name  # e.g. sft-0.6b-s0 / dpo-0.6b-s0
        if not (ad / "adapter_config.json").exists():
            continue
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype).cuda()
        model = PeftModel.from_pretrained(base, str(ad))
        for s, rows in splits.items():
            np.save(root / f"{s}_{name}.npy", score_rows(model, tok, rows, serve, cfg.max_len))
        found[name] = str(ad)
        del model, base
        torch.cuda.empty_cache()
    res = {"adapters": found, "base_model": cfg.base_model, "seed": cfg.seed}
    (root / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return res


def run_export(  # pragma: no cover - heavy
    cfg: TrainConfig,
    serve: ServeConfig,
    adapters_root: Path,
    which: str = "sft-0.6b-s0",
    variants: tuple[str, ...] = ("fp32", "w8"),
    parity_rows: int = 0,
) -> dict[str, Any]:
    """Merge (or take the untuned base for ``which="base"``) -> slim ONNX variants -> calibration.json.

    ``parity_rows`` > 0 scores that many gold_test rows (-1: all) with every exported variant on this CPU and
    saves the 4-way logits (``gold_<variant>_logits.npy``) for the quality-parity check against torch; the
    mean per-row time is recorded as ``ms_per_row``. Serving latency is measured on the deployed service instead.
    """
    import os
    import time

    import numpy as np
    import onnxruntime

    from breachlab.serve.onnx_backend import load_guard
    from breachlab.train.export_onnx import (
        VARIANT_FILES,
        calibration_for,
        export_onnx,
        load_results,
        manifest,
    )
    from breachlab.train.prompts import encode_prompt

    adapter = None
    if which != "base":
        adapter = next(p for p in adapters_root.rglob("adapter") if p.parent.name == which)
    out = cfg.output_dir / f"export-{which}"
    export_onnx(cfg.base_model, adapter, serve.token_ids(), out, variants)
    cal = calibration_for(which, load_results(adapters_root))
    (out / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    threads = os.cpu_count() or 4
    res: dict[str, Any] = {
        "which": which,
        "cpu_threads": threads,
        "onnxruntime": onnxruntime.__version__,
        "calibration": cal,
        "parity_ms_per_row": {},
    }
    if parity_rows:
        gold = read_rows(cfg.data_dir / "gold_test.jsonl")
        sample = gold if parity_rows < 0 else gold[:parity_rows]
        for variant in variants:
            fname = VARIANT_FILES[variant]
            if not (out / fname).exists():
                res["parity_ms_per_row"][variant] = "file missing (see w8_error.txt)"
                continue
            model = load_guard(out, serve, which, threads=threads, file=fname)
            t0 = time.perf_counter()
            lg = np.stack(
                [
                    model.logits_fn(
                        encode_prompt(model.encode, r.text[:500], model.words, "input", 512)
                    )
                    for r in sample
                ]
            )
            res["parity_ms_per_row"][variant] = (time.perf_counter() - t0) * 1000 / len(sample)
            np.save(out / f"gold_{variant}_logits.npy", lg)
    res["files"] = manifest(out)
    (out / "export.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res
